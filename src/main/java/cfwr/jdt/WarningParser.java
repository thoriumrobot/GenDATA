package cfwr.jdt;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * JDT-based warning parser for Checker Framework warnings.
 * Replaces regex-based parsing with more robust structured parsing.
 */
public class WarningParser {
    
    // Multiple patterns for different warning formats
    private static final Pattern[] WARNING_PATTERNS = {
        // Standard Checker Framework format: file:line:col: compiler.err.proc.messager: [checker] message
        Pattern.compile("^(.+\\.java):(\\d+):(\\d+):\\s*(compiler\\.(warn|err)\\.proc\\.messager):\\s*\\[(.+?)\\]\\s*(.*)$"),
        
        // Simple format: file:line: error/warning: message
        Pattern.compile("^(.*\\.java):(\\d+):\\s*(error|warning):\\s*(.*)$"),
        
        // Extended format: file:line:col: message
        Pattern.compile("^(.*\\.java):(\\d+):(\\d+):\\s*(.*)$"),
        
        // Minimal format: file:line: message
        Pattern.compile("^(.*\\.java):(\\d+):\\s*(.*)$")
    };
    
    public List<WarningInfo> parseWarnings(String warningsFile) throws IOException {
        List<WarningInfo> warnings = new ArrayList<>();
        
        if (!Files.exists(Paths.get(warningsFile))) {
            throw new FileNotFoundException("Warnings file not found: " + warningsFile);
        }
        
        List<String> lines = Files.readAllLines(Paths.get(warningsFile));
        
        for (int lineNum = 0; lineNum < lines.size(); lineNum++) {
            String line = lines.get(lineNum).trim();
            
            if (line.isEmpty() || line.startsWith("#")) {
                continue; // Skip empty lines and comments
            }
            
            WarningInfo warning = parseWarningLine(line, lineNum + 1);
            if (warning != null) {
                warnings.add(warning);
            }
        }
        
        return warnings;
    }
    
    private WarningInfo parseWarningLine(String line, int lineNumber) {
        for (int i = 0; i < WARNING_PATTERNS.length; i++) {
            Pattern pattern = WARNING_PATTERNS[i];
            Matcher matcher = pattern.matcher(line);
            
            if (matcher.matches()) {
                return createWarningInfo(matcher, i, lineNumber);
            }
        }
        
        // If no pattern matches, create a generic warning info
        return createGenericWarningInfo(line, lineNumber);
    }
    
    private WarningInfo createWarningInfo(Matcher matcher, int patternIndex, int lineNumber) {
        WarningInfo warning = new WarningInfo();
        warning.setLineNumber(lineNumber);
        
        switch (patternIndex) {
            case 0: // Standard CF format
                warning.setFilePath(matcher.group(1));
                warning.setLine(Integer.parseInt(matcher.group(2)));
                warning.setColumn(Integer.parseInt(matcher.group(3)));
                warning.setSeverity(matcher.group(4).equals("err") ? "error" : "warning");
                warning.setChecker(matcher.group(5));
                warning.setMessage(matcher.group(6));
                break;
                
            case 1: // Simple format
                warning.setFilePath(matcher.group(1));
                warning.setLine(Integer.parseInt(matcher.group(2)));
                warning.setColumn(0);
                warning.setSeverity(matcher.group(3));
                warning.setChecker("unknown");
                warning.setMessage(matcher.group(4));
                break;
                
            case 2: // Extended format
                warning.setFilePath(matcher.group(1));
                warning.setLine(Integer.parseInt(matcher.group(2)));
                warning.setColumn(Integer.parseInt(matcher.group(3)));
                warning.setSeverity("warning");
                warning.setChecker("unknown");
                warning.setMessage(matcher.group(4));
                break;
                
            case 3: // Minimal format
                warning.setFilePath(matcher.group(1));
                warning.setLine(Integer.parseInt(matcher.group(2)));
                warning.setColumn(0);
                warning.setSeverity("warning");
                warning.setChecker("unknown");
                warning.setMessage(matcher.group(3));
                break;
        }
        
        return warning;
    }
    
    private WarningInfo createGenericWarningInfo(String line, int lineNumber) {
        WarningInfo warning = new WarningInfo();
        warning.setLineNumber(lineNumber);
        warning.setFilePath("unknown");
        warning.setLine(0);
        warning.setColumn(0);
        warning.setSeverity("unknown");
        warning.setChecker("unknown");
        warning.setMessage(line);
        return warning;
    }
    
    /**
     * Warning information extracted from Checker Framework output.
     */
    public static class WarningInfo {
        private int lineNumber; // Line number in the warnings file
        private String filePath;
        private int line;
        private int column;
        private String severity;
        private String checker;
        private String message;
        
        // Getters and setters
        public int getLineNumber() { return lineNumber; }
        public void setLineNumber(int lineNumber) { this.lineNumber = lineNumber; }
        
        public String getFilePath() { return filePath; }
        public void setFilePath(String filePath) { this.filePath = filePath; }
        
        public int getLine() { return line; }
        public void setLine(int line) { this.line = line; }
        
        public int getColumn() { return column; }
        public void setColumn(int column) { this.column = column; }
        
        public String getSeverity() { return severity; }
        public void setSeverity(String severity) { this.severity = severity; }
        
        public String getChecker() { return checker; }
        public void setChecker(String checker) { this.checker = checker; }
        
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        
        @Override
        public String toString() {
            return String.format("WarningInfo{file='%s', line=%d, col=%d, severity='%s', checker='%s', message='%s'}", 
                filePath, line, column, severity, checker, message);
        }
    }
}
