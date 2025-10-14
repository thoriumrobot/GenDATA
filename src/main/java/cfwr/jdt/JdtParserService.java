package cfwr.jdt;

import cfwr.jdt.CodeLocationAnalyzer;
import cfwr.jdt.WarningParser;
import cfwr.jdt.IdentifierExtractor;
import cfwr.jdt.util.JsonOutput;
import cfwr.jdt.WarningParser.WarningInfo;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Main CLI service for JDT-based parsing operations.
 * Replaces regex-based parsing in Python components with robust AST parsing.
 */
public class JdtParserService {
    
    public static void main(String[] args) {
        if (args.length < 2) {
            printUsage();
            System.exit(1);
        }
        
        try {
            Map<String, String> params = parseArgs(args);
            String operation = params.get("--operation");
            
            if (operation == null) {
                System.err.println("Error: --operation is required");
                printUsage();
                System.exit(1);
            }
            
            switch (operation) {
                case "parse-code-locations":
                    handleParseCodeLocations(params);
                    break;
                case "parse-warnings":
                    handleParseWarnings(params);
                    break;
                case "parse-identifiers":
                    handleParseIdentifiers(params);
                    break;
                case "validate-syntax":
                    handleValidateSyntax(params);
                    break;
                default:
                    System.err.println("Error: Unknown operation: " + operation);
                    printUsage();
                    System.exit(1);
            }
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static void printUsage() {
        System.err.println("Usage: java cfwr.jdt.JdtParserService --operation <op> [options]");
        System.err.println();
        System.err.println("Operations:");
        System.err.println("  parse-code-locations --input <file> --output <file>");
        System.err.println("    Parse Java file and extract code locations (methods, classes, etc.)");
        System.err.println();
        System.err.println("  parse-warnings --input <file> --output <file>");
        System.err.println("    Parse Checker Framework warnings file");
        System.err.println();
        System.err.println("  parse-identifiers --input <file> --output <file>");
        System.err.println("    Extract identifiers from Java code");
        System.err.println();
        System.err.println("  validate-syntax --input <file> --output <file>");
        System.err.println("    Validate Java syntax and return result");
    }
    
    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> params = new HashMap<>();
        
        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--")) {
                if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    params.put(args[i], args[i + 1]);
                    i++;
                } else {
                    params.put(args[i], "true");
                }
            }
        }
        
        return params;
    }
    
    private static void handleParseCodeLocations(Map<String, String> params) throws IOException {
        String inputFile = requireParam(params, "--input");
        String outputFile = requireParam(params, "--output");
        
        String javaCode = Files.readString(Paths.get(inputFile));
        
        CodeLocationAnalyzer analyzer = new CodeLocationAnalyzer();
        List<CodeLocation> locations = analyzer.analyzeCode(javaCode);
        
        String jsonOutput = JsonOutput.toJson(locations);
        Files.writeString(Paths.get(outputFile), jsonOutput);
        
        System.out.println("Parsed " + locations.size() + " code locations");
    }
    
    private static void handleParseWarnings(Map<String, String> params) throws IOException {
        String inputFile = requireParam(params, "--input");
        String outputFile = requireParam(params, "--output");
        
        WarningParser parser = new WarningParser();
        List<WarningInfo> warnings = parser.parseWarnings(inputFile);
        
        String jsonOutput = JsonOutput.toJson(warnings);
        Files.writeString(Paths.get(outputFile), jsonOutput);
        
        System.out.println("Parsed " + warnings.size() + " warnings");
    }
    
    private static void handleParseIdentifiers(Map<String, String> params) throws IOException {
        String inputFile = requireParam(params, "--input");
        String outputFile = requireParam(params, "--output");
        
        String javaCode = Files.readString(Paths.get(inputFile));
        
        IdentifierExtractor extractor = new IdentifierExtractor();
        Map<String, List<String>> identifiers = extractor.extractIdentifiers(javaCode);
        
        String jsonOutput = JsonOutput.toJson(identifiers);
        Files.writeString(Paths.get(outputFile), jsonOutput);
        
        System.out.println("Extracted identifiers: " + identifiers.keySet());
    }
    
    private static void handleValidateSyntax(Map<String, String> params) throws IOException {
        String inputFile = requireParam(params, "--input");
        String outputFile = requireParam(params, "--output");
        
        String javaCode = Files.readString(Paths.get(inputFile));
        
        CodeLocationAnalyzer analyzer = new CodeLocationAnalyzer();
        boolean isValid = analyzer.validateSyntax(javaCode);
        
        Map<String, Object> result = new HashMap<>();
        result.put("valid", isValid);
        result.put("message", isValid ? "Syntax is valid" : "Syntax errors found");
        
        String jsonOutput = JsonOutput.toJson(result);
        Files.writeString(Paths.get(outputFile), jsonOutput);
        
        System.out.println("Syntax validation: " + (isValid ? "PASSED" : "FAILED"));
    }
    
    private static String requireParam(Map<String, String> params, String key) {
        String value = params.get(key);
        if (value == null) {
            throw new IllegalArgumentException("Missing required parameter: " + key);
        }
        return value;
    }
}
