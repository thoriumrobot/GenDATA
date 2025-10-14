package cfwr.jdt;

import java.util.Map;
import java.util.Set;

/**
 * Represents a code location identified by JDT parser.
 * Replaces the Python CodeLocation dataclass for Java compatibility.
 */
public class CodeLocation {
    private int lineStart;
    private int lineEnd;
    private int columnStart;
    private int columnEnd;
    private String locationType;
    private Map<String, Object> context;
    private String codeSnippet;
    private Set<String> applicableTransformations;
    
    public CodeLocation() {
        // Default constructor for Jackson
    }
    
    public CodeLocation(int lineStart, int lineEnd, int columnStart, int columnEnd,
                       String locationType, Map<String, Object> context,
                       String codeSnippet, Set<String> applicableTransformations) {
        this.lineStart = lineStart;
        this.lineEnd = lineEnd;
        this.columnStart = columnStart;
        this.columnEnd = columnEnd;
        this.locationType = locationType;
        this.context = context;
        this.codeSnippet = codeSnippet;
        this.applicableTransformations = applicableTransformations;
    }
    
    // Getters and setters
    public int getLineStart() { return lineStart; }
    public void setLineStart(int lineStart) { this.lineStart = lineStart; }
    
    public int getLineEnd() { return lineEnd; }
    public void setLineEnd(int lineEnd) { this.lineEnd = lineEnd; }
    
    public int getColumnStart() { return columnStart; }
    public void setColumnStart(int columnStart) { this.columnStart = columnStart; }
    
    public int getColumnEnd() { return columnEnd; }
    public void setColumnEnd(int columnEnd) { this.columnEnd = columnEnd; }
    
    public String getLocationType() { return locationType; }
    public void setLocationType(String locationType) { this.locationType = locationType; }
    
    public Map<String, Object> getContext() { return context; }
    public void setContext(Map<String, Object> context) { this.context = context; }
    
    public String getCodeSnippet() { return codeSnippet; }
    public void setCodeSnippet(String codeSnippet) { this.codeSnippet = codeSnippet; }
    
    public Set<String> getApplicableTransformations() { return applicableTransformations; }
    public void setApplicableTransformations(Set<String> applicableTransformations) { 
        this.applicableTransformations = applicableTransformations; 
    }
    
    @Override
    public String toString() {
        return String.format("CodeLocation{type='%s', lines=%d-%d, cols=%d-%d, transformations=%s}", 
            locationType, lineStart, lineEnd, columnStart, columnEnd, applicableTransformations);
    }
}
