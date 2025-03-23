const express = require('express');
const client = require('prom-client');

const app = express();
const port = 8080;

// Create a Registry for Prometheus
const register = new client.Registry();

// Define a basic metric
const httpRequestCounter = new client.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
});

register.registerMetric(httpRequestCounter);

// Middleware to count requests
app.use((req, res, next) => {
  httpRequestCounter.inc();
  next();
});

// Expose metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});

// Root endpoint
app.get('/', (req, res) => {
  res.send('Hello, this is Node.js with Prometheus metrics!');
});

app.listen(port, () => {
  console.log(`App listening at http://localhost:${port}`);
});
