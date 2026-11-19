package ht.mbds.salimbenzeragah.tp4test;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test3 {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    // Méthode utilitaire pour ingérer un document et créer son ContentRetriever
    private static ContentRetriever creerRetriever(String resourceName, EmbeddingModel embeddingModel) {
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument(resourceName, parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    public static void main(String[] args) {
        configureLogger();

        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("Erreur : la variable d'environnement GEMINI_KEY n'est pas définie !");
            return;
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.8-flash")
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();

        // 1. Création des deux retrievers (2 bases vectorielles séparées)
        ContentRetriever retrieverCours = creerRetriever("rag.pdf", embeddingModel);
        ContentRetriever retrieverReglement = creerRetriever("reglement.txt", embeddingModel);

        // 2. Association de chaque retriever à une description claire pour le routeur
        Map<ContentRetriever, String> retrieverToDescription = new HashMap<>();
        retrieverToDescription.put(retrieverCours,
                "Support de cours d'informatique sur le RAG (Retrieval-Augmented Generation), les embeddings et le fine-tuning des LLM");
        retrieverToDescription.put(retrieverReglement,
                "Règlement intérieur de l'entreprise TechMaroc concernant les horaires, le télétravail, les congés payés et la sécurité informatique");

        // 3. Configuration du routeur (ne route pas si la question ne concerne aucune source)
        QueryRouter queryRouter = LanguageModelQueryRouter.builder()
                .chatModel(model)
                .retrieverToDescription(retrieverToDescription)
                .fallbackStrategy(LanguageModelQueryRouter.FallbackStrategy.DO_NOT_ROUTE)
                .build();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(chatMemory)
                .build();

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question (ou 'fin' pour quitter) :");
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