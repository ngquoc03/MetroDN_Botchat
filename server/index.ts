import "dotenv/config";

// Import Express framework for creating web server
import express, { Express, Request, Response } from "express";

// Import MongoDB client for database connection
import { MongoClient } from "mongodb";

// Import our custom AI agent function
import { callAgent } from "./agent";

// Import CORS middleware for handling cross-origin requests
import cors from "cors";

// Create Express application instance
const app: Express = express();

// Enable CORS for all routes (allows frontend to call this API)
app.use(cors());

// Enable JSON parsing for incoming requests (req.body will contain parsed JSON)
app.use(express.json());

// 🔎 Debug: Log MongoDB URI to check if .env is working
console.log("MongoDB URI from ENV:", process.env.MONGODB_ATLAS_URI);

// Get MongoDB URI from environment variables
const uri = process.env.MONGODB_ATLAS_URI;

// If no URI is found, throw an error and stop the server
if (!uri) {
  throw new Error("❌ Missing MONGODB_ATLAS_URI in .env file");
}

// Create MongoDB client using connection string
const client = new MongoClient(uri);

// Async function to initialize and start the server
async function startServer() {
  try {
    // Establish connection to MongoDB Atlas
    await client.connect();
    // Ping MongoDB to verify connection is working
    await client.db("admin").command({ ping: 1 });
    // Log successful connection
    console.log("✅ You successfully connected to MongoDB!");

    // Define root endpoint (GET /) - simple health check
    app.get("/", (req: Request, res: Response) => {
      res.send("LangGraph Agent Server");
    });

    // Define endpoint for starting new conversations (POST /chat)
    app.post("/chat", async (req: Request, res: Response) => {
      const initialMessage = req.body.message;
      const threadId = Date.now().toString();
      console.log("Incoming message:", initialMessage);

      try {
        const response = await callAgent(client, initialMessage, threadId);
        res.json({ threadId, response });
      } catch (error) {
        console.error("Error starting conversation:", error);
        res.status(500).json({ error: "Internal server error" });
      }
    });

    // Define endpoint for continuing existing conversations (POST /chat/:threadId)
    app.post("/chat/:threadId", async (req: Request, res: Response) => {
      const { threadId } = req.params;
      const { message } = req.body;

      try {
        const response = await callAgent(client, message, threadId);
        res.json({ response });
      } catch (error) {
        console.error("Error in chat:", error);
        res.status(500).json({ error: "Internal server error" });
      }
    });

    // Get port from environment variable or default to 8000
    const PORT = process.env.PORT || 8000;

    app.listen(PORT, () => {
      console.log(`🚀 Server running on port ${PORT}`);
    });
  } catch (error) {
    console.error("❌ Error connecting to MongoDB:", error);
    process.exit(1);
  }
}

// Start the server
startServer();
