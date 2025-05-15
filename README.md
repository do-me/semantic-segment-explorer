# Semantic Segment Explorer

**Live Demo: https://do-me.github.io/semantic-segment-explorer/**

In-browser tool to explore semantic similarity by generating and querying overlapping text segments using Transformers.js. This application allows you to input a source text, which is then broken down into numerous overlapping segments. Each unique segment is embedded using `minishlab/potion-retrieval-32M`, and you can then query these segments to find those most semantically similar to your query.

The main motivation behind this app is to experiment with different text chunking/segmentation strategies and observe how the semantic similarity results vary, especially with segments of different lengths.

## 📑 Key Features

*   **Flexible Text Input:** Paste any source text to analyze.
*   **Advanced Segmentation:**
    *   **With Sentence Boundaries (Default):** Text is split into sentences, and segments (all contiguous word combinations) are generated *within* each sentence.
    *   **Without Sentence Boundaries:** Segments are generated from all contiguous word combinations across the entire text.
*   **In-Browser Embeddings:** Uses [Hugging Face Transformers.js](https://github.com/huggingface/transformers.js/) with the [minishlab/potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) model to generate embeddings directly in the user's browser. No server-side processing needed for the core AI!
*   **Semantic Querying:** Find text segments most semantically similar to your input query.
*   **Query-As-You-Type:** (Optional) Get instant search results as you type your query.

## 🤔 How It Works

1.  **Input Text:** The user provides a source text.
2.  **Segmentation:**
    *   The text is processed based on the "Use Sentence Boundaries" setting.
    *   **With Sentence Boundaries:** The text is first split into individual sentences. Then, for each sentence, all possible contiguous sub-sequences of words are generated as segments.
    *   **Without Sentence Boundaries:** All possible contiguous sub-sequences of words are generated from the entire input text.
    *   Duplicate segments are removed.
3.  **Embedding:** Each unique segment is converted into a numerical vector (embedding) using the [minishlab/potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) model running via Transformers.js in the browser.
4.  **Indexing:** These embeddings are stored locally in the browser's memory.
5.  **Querying:**
    *   The user inputs a query.
    *   The query is also embedded using the same model.
    *   The cosine similarity between the query embedding and all indexed segment embeddings is calculated.
    *   The top N most similar segments are displayed as results.

### Segment Generation Complexity:

*   **With Sentence Boundaries (default):** The text is first split into S sentences. Segments are generated within each sentence. If N<sub>s</sub> is the average number of words per sentence, the number of unique segments is roughly Σ(N<sub>s,i</sub>\*(N<sub>s,i</sub>+1)/2) for each sentence *i* (though duplicates across sentences or within are removed). This is generally less than O(N<sup>2</sup>) for the total number of words N.
*   **Without Sentence Boundaries:** The number of potential segments grows quadratically with the total number of words (N) in your source text, approximately N \* (N+1) / 2. After removing duplicates, the actual count may be lower but can still be substantial. This is O(N<sup>2</sup>) in terms of combinations generated.
    *   **Warning:** For long texts (e.g., >1000 words), using this option can lead to a very large number of segments, potentially crashing the browser tab due to memory or processing limits.

## 🛠️ Technical Stack

*   **Frontend:** HTML, [Tailwind CSS](https://tailwindcss.com/)
*   **JavaScript:** Vanilla JS (ES Modules)
*   **Machine Learning:** [Hugging Face Transformers.js](https://huggingface.co/docs/transformers.js)
*   **Embedding Model:** [minishlab/potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) (a compact and efficient model suitable for in-browser use)

## 🚀 Getting Started / How to Use

1.  **Visit the Live Demo:** [YOUR_DEMO_URL_HERE]
2.  **Input Source Text:** Paste your text into the main textarea.
3.  **Choose Segmentation Strategy:** Decide whether to use sentence boundaries (checked by default, recommended for longer texts).
4.  **Generate & Index:** Click the "Generate & Index Segments" button. Wait for the processing to complete (progress will be shown).
5.  **Query:** Once indexing is done, the query section will appear. Type your search query.
6.  **Search:** Click "Search" or enable "Query-As-You-Type" for instant results.
7.  **View Results:** Semantically similar segments from your source text will be displayed.

## 💡 Motivation & Related Work

This project was primarily created to experiment with different text segmentation (chunking) techniques for semantic search. Segment length and boundaries do affect retrieval quality obviously but it's cool to see how sometimes longer segments are more similar to a query.

If you're interested in semantic search or similar in-browser AI applications, you might also like:

*   **[SemanticFinder](https://do-me.github.io/SemanticFinder/):** Another in-browser semantic search tool focusing on sentence-level similarity.
*   **[Guerilla Semantic Search Tutorial](https://geo.rocks/post/semantic-search-tutorial/#foreword-guerilla-semantic-search):** An article discussing approaches to client-side semantic search.
*   **[My other semantic search projects on GitHub](https://github.com/do-me?tab=repositories&q=semantic&type=&language=&sort=):** A collection of related experiments and tools.

## 🤝 Contributing

While this demo is mainly experimental I don't intend on developing it further. Still, your contributions are more than welcome! 

## 📜 License

Distributed under the MIT License.

## 🙏 Acknowledgements

*   Xenova and [Hugging Face](https://huggingface.co/) for their incredible `transformers.js` library and model hosting.
*   The creators of the [minishlab/potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) for creating super fast static embeddings
*   [Tailwind CSS](https://tailwindcss.com/)
