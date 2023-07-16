## sinewCHAT

![sinew](https://github.com/EveryOneIsGross/sinewCHAT/assets/23621140/fcfd3366-76c3-4a89-945f-4a148c87b7ff)


# a neuron-based chatbot for sentiment alignment. 

This project implements a neural conversational agent using OpenAI's language model. The agent is designed to generate responses, analyze sentiment, extract keywords, and provide summaries based on user input. The architecture of the sinewCHAT project is based on a neural model, which is an abstraction of how a biological brain works. In this model, each chatbot instance is treated as a neuron. Here's a high-level overview of the architecture:

OpenAIAgent: This is the base class that interacts with the OpenAI API. It is responsible for generating responses and summaries from a given prompt using openai api.

Neuron: This class represents a neuron in the neural model. Each Neuron instance uses an OpenAIAgent to generate responses and calculate sentiment scores. It also extracts relevant keywords from the responses.

NeuralModel: This class uses multiple instances of the Neuron class to process input. It calculates the sentiment of the input and adjusts the responses of the neurons based on the sentiment scores. It also generates a summary of the adjusted responses.

Main Loop: This is the entry point of the program. It prompts the user for input, processes the input using the NeuralModel, and prints the output and summary. The loop continues to prompt the user for input until the user decides to stop.

This architecture allows the system to generate enriched and positive-weighted responses by using multiple chatbot instances (neurons) and adjusting their responses based on the sentiment of the input.

## Features

- Uses an instanced q&a chatbot.
- Performs sentiment analysis using the NLTK library.
- Extracts keywords from responses using the RAKE library.
- Provides summaries of the conversation based on adjusted responses.
- Implements a neural model with multiple processing units (neurons) to collectively analyze and respond to user input.
- Updates attention and feedback weights based on user sentiment and aggregated sentiment scores from the neurons.

## Components

### `chatBOT`

- Acts as the interface for interacting with OpenAI.
- Generates responses based on user prompts.
- Supports chat-based conversations using the OpenAI Chat Completion API.

### `Neuron`

- Represents a processing unit within the neural model.
- Analyzes sentiment using the NLTK library.
- Extracts relevant keywords from responses using the RAKE library.
- Keeps track of responses, sentiment scores, and keywords.
- Implements a mechanism to find the last positive response.

### `NeuralModel`

- Orchestrates the behavior of the neurons and the overall decision-making process.
- Processes user input and distributes attention weights among the neurons.
- Updates attention and feedback weights based on user sentiment and aggregated sentiment scores from the neurons.
- Generates summaries based on the adjusted responses from the neurons.
- Saves conversation data in a JSON file.

## Usage

1. Set up the necessary dependencies: numpy, nltk, rake-nltk, and OpenAI Python library.
2. Create an instance of the `OpenAIAgent` class, providing the appropriate OpenAI model.
3. Create an instance of the `NeuralModel` class, passing the `OpenAIAgent` instance.
4. Interact with the agent by calling the `process_input` method of the `NeuralModel` instance, providing user prompts.
5. View the generated output, summary, and summary keyword.

## Limitations and Future Enhancements

- The current implementation uses a simplified neural model and may not capture complex conversational dynamics.
- The keyword extraction process could be further improved for better relevance and accuracy.
- The feedback mechanism for updating weights is simple and may benefit from more advanced algorithms.

## License

This project is licensed under the [MIT License](LICENSE).


