## sinewCHAT

![sinew](https://github.com/EveryOneIsGross/sinewCHAT/assets/23621140/fcfd3366-76c3-4a89-945f-4a148c87b7ff)


# a pseudo-neuralnet chatbot

This project implements a neural conversational agent using OpenAI's language model. The agent is designed to generate responses, analyze sentiment, extract keywords, and provide summaries based on user input.

## Features

- Uses an instanced q&a chatbot.
- Performs sentiment analysis using the NLTK library.
- Extracts keywords from responses using the RAKE library.
- Provides summaries of the conversation based on adjusted responses.
- Implements a neural model with multiple processing units (neurons) to collectively analyze and respond to user input.
- Updates attention and feedback weights based on user sentiment and aggregated sentiment scores from the neurons.

## Components

### `chatBOT`

- Acts as the interface for interacting with OpenAI's language model.
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


