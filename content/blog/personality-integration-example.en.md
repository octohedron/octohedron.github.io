---
title: "Personality insights: Integrating IBM Watson example"
date: 2019-04-14T17:30:40+06:00
image: images/blog/watson-integration.jpg
author: CEO
---

### How we integrated Watson Personality Insights to analyze hundreds of millions of users

---

One of our biggest clients needed to **pair social media influencers with brands**, and a great way to use **A.I.** for that purpose would be to use Watson's personality insights to figure out which influencers would be a good pair for each brand.

For example, Red Bull is associated with extreme sports, and it would make sense to match it with someone that is extroverted, thriving for adventure, very active, etc.

All we had to do was integrate our client's data pipeline with Watson's API, and now the client can do queries in their database to find out which profiles would be a better match than others.

This sounds easy in practice but there's a lot of things that could go wrong and implementation details I would like to tell you about.

#### Step 1 - Designing our pipeline

We need to define a way in which we can do this fast, but asynchronously. To make a good Watson analysis, we need as much data as possible, and that requires a lot of requests, which can take some time. So, the best way to do this is to use a job state pattern.

The way it works is simple, as simple as possible.

+ 1. A request of analysis for a user is made to the API.
+ 2. The API sets the status as "requested" for the user.
+ 3. The API calls a different service and schedules the request of the data for the analysis.
+ 4. The data requester service downloads the data.
+ 5. The data is cleaned up of hashtags and mentions symbols that don't add up to the analysis.
+ 5. A new request to Watson's API is made with the clean data.
+ 6. The analysis results from the Watson API are stored in the server.
+ 7. The state of the job is updated.

---

After all steps are complete, when a new request is made to the same endpoint, instead of scheduling a job, it should return the analysis results.


#### Step 2 - The technology

To implement this type of pipeline, we will need a few things to make it easier and simpler.

+ 1. An API to handle the state, return results, schedule jobs, etc.
+ 2. A service to request, clean and and save analysis results.
+ 3. Something to glue everything together, i.e. a global shared state.

---

For this project we chose 3 technologies that gave us the advantage for a few reasons.

+ 1. We implemented the API in **Go**, because it's perfect for that type of service.
+ 2. We implemented the data requesting service in **Python** because of all the solutions already made for it and quick development time.
+ 3. We chose **redis** as a global shared state because of performance and features.

---

By choosing these technologies, we managed to implement the whole pipeline in just a few hours, by leveraging the best and fastest to implement technologies for each step.

#### Step 3 - State handling and orchestration implementation

We implement a simple state handling and orchestration API in Go.

The concept is simple, as simple as possible. When an analysis request is made to the API, the status of the job is checked in the global state. If it's found to be completed, it means that the data was already requested, processed, analyzed and the final result saved in the database, in which case we just need to fetch it and return it. Otherwise it will schedule it.

``` go
// Analyze a user making a POST request to /analyze with raw payload "username"
func analyze(w http.ResponseWriter, r *http.Request) {
  // Only allow POST requests in this endpoint
  if r.Method != "POST" {
    http.Error(w, fmt.Sprintf("{ \"error\": \"%s\" }", "Method not allowed"), 405)
    return
  }
  // Allocate a user string
  var user string
  // Read the request
  body, err := ioutil.ReadAll(r.Body)
  // Check there's no errors reading the request
  if err != nil {
    http.Error(w, fmt.Sprintf("{ \"error\": \"%s\" }", "Malformed request"), 403)
    return
  }
  // Unmarshall the username into the user string
  err = json.Unmarshal(body, &user)
  // Get handle to redis connection pool
  conn := POOL.Get()
  // Check the current status
  status, err := redis.String(conn.Do("hget", "analysis:"+user, "status"))
  // If there's an error here, it could mean that there was a problem with redis
  // or that it's a new analysis request job, we will assume that for simplicity
  if err != nil {
    status = "not_requested"
  }
  // We handle each state with a simple switch statement
  switch status {
  case "not_requested":
    // Request the data and analysis
    go fetchData(user, "analysis", "username")
    // Set the status as scheduled
    _, err = redis.Int(conn.Do("HSET", "analysis:"+user, "status", "scheduled"))
    // Check if there was a problem when setting the redis status
    LogPanic(err, "Error performing HSET in redis")
    // respond with scheduled
    w.Write([]byte("scheduled"))
  case "completed":
    // fetch the analysis data from the database and return it
    w.Write([]byte("..."))
```

We're leaving out the part of fetching the data from the database because that's quite simple, we just want to illustrate the concept of a simple integration.


#### Step 4 - A data requesting, processing and analyzing service

For these steps of the pipeline, we use a of really powerful python tool that it's extremely simple and fast to set up and configure, **scrapy**.

Scrapy can be easily orchestrated with **scrapyd**. It's as simple as making a request to the scrapyd service and it will schedule a job.

After we request the data, the pipeline is very simple


``` python

class AllUserDataPipeline(object):
    """This pipeline will perform a Watson analysis"""
    collection_name = 'user_analysis'

    def __init__(self, mongo_uri, mongo_db, redis_host, redis_port):
      """Set up the database connections parameters"""
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.redis_host = redis_host
        self.redis_port = redis_port

    def open_spider(self, spider):
        """Initialize the database connections when the spider is opened"""
        # Initialize the database connections...
        # Set job to running when we start the crawl
        self.redis_conn.hset(
            "analysis:" + spider.username,
            "status", "running")

    def close_spider(self, spider):
        """Perform Watson analysis and update status accordingly"""
        # Get data from the spider (no data is saved)
        text = spider.data
        # Remove hashtags and mentions symbols
        text = text.replace("#", " ")
        text = text.replace("@", " ")
        # If we have enough text for a decent analysis
        if len(text) > 200:
            # Initialize request with Watson URL
            res = requests.post(
                "https://gateway-fra.watsonplatform.net/personality-insights" +
                "/api/v3/profile?version=2017-10-13&raw_scores=true",
                auth=(
                    "api_key", "your_key"),
                headers={
                    "content-type": "text/plain", "Accept": "application/json"},
                data=text.encode("utf-8")
            )
            # Save result
            self.db[self.collection_name] \
                .update({'username': spider.username},
                        {"$set": {"watson_result": json.loads(res.text)}}, True)
        # Set status
        self.redis_conn.hset("analysis:" + spider.username, "status", "completed")
        # Close database connection
        self.client.close()
```

As you can see, with just a few lines of python we can process the data, request an analysis and save the results.

Now, what we have is a **scheduled system** that can process a virtually unlimited amount of requests as long as we have enough hardware to run it. It will automatically allocate as much schedule jobs as we can for our hardware set up.

For example, if we requested 10 million analysis at once, and we can only do 20 at a time because we don't have enough hardware resources, it will only have at most 20 jobs running at once, but will eventually complete all of them.

*DISCLAIMER: No data is stored in our client's server, each analysis is performed on the fly.*