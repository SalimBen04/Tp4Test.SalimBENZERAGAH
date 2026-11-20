package ht.mbds.salimbenzeragah.tp4test;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test5 {

    interface Assistant {
        String chat(String userMessage);
    }

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) throws URISyntaxException {
        configureLogger();

        String geminiApiKey = System.getenv("GEMINI_KEY");
        String tavilyApiKey = System.getenv("TAVILY_KEY");

        if (geminiApiKey == null || geminiApiKey.isEmpty() || tavilyApiKey == null || tavilyApiKey.isEmpty()) {
            System.err.println("Erreur : GEMINI_KEY et TAVILY_KEY doivent être définies dans les variables d'environnement !");
            return;
        }

        // 1. Modèle de langage Gemini
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-3.8-flash")
                .logRequestsAndResponses(true)
                .build();

        // 2. Ingestion du PDF local (rag.pdf)
        URL resourceUrl = Test5.class.getResource("/rag.pdf");
        if (resourceUrl == null) {
            System.err.println("Fichier rag.pdf introuvable dans resources !");
            return;
        }
        Path documentPath = Paths.get(resourceUrl.toURI());
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // Retriever 1 : Base vectorielle locale (cours RAG)
        ContentRetriever retrieverPdf = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // 3. Retriever 2 : Recherche Web via Tavily
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyApiKey)
                .build();

        ContentRetriever retrieverWeb = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .maxResults(3)
                .build();

        // 4. Routage intelligent via LanguageModelQueryRouter (Bonus)
        Map<ContentRetriever, String> retrieverToDescription = new HashMap<>();
        retrieverToDescription.put(
                retrieverPdf,
                "Support de cours sur le RAG (Retrieval-Augmented Generation), les LLMs, les embeddings et LangChain4j"
        );
        retrieverToDescription.put(
                retrieverWeb,
                "Recherche sur le Web pour les actualités récentes, les événements ou tout sujet extérieur au cours sur le RAG"
        );

        QueryRouter queryRouter = new LanguageModelQueryRouter(model, retrieverToDescription);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 5. Création de l'assistant avec mémoire
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(chatMemory)
                .build();

        // 6. Boucle conversationnelle
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.print("Posez votre question (ou 'fin' pour quitter) : ");
                String question = scanner.nextLine();
                if ("fin".equalsIgnoreCase(question.trim())) {
                    break;
                }
                if (question.isBlank()) {
                    continue;
                }
                String reponse = assistant.chat(question);
                System.out.println("Réponse :\n" + reponse);
            }
        }
    }
}