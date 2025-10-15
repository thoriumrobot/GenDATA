package cfwr.jdt;

import java.util.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Comprehensive diagnostics system for transformation failures and analysis.
 * Captures detailed information about transformation decisions, failures, and performance.
 */
public class TransformationDiagnostics {
    
    private final String sessionId;
    private final LocalDateTime startTime;
    private final Map<String, TransformationRecord> transformations;
    private final List<DiagnosticEntry> diagnostics;
    private final PerformanceMetrics performance;
    
    public TransformationDiagnostics() {
        this.sessionId = generateSessionId();
        this.startTime = LocalDateTime.now();
        this.transformations = new HashMap<>();
        this.diagnostics = new ArrayList<>();
        this.performance = new PerformanceMetrics();
    }
    
    /**
     * Record the start of a transformation.
     */
    public void recordTransformationStart(String transformation, String mode, String originalCode) {
        TransformationRecord record = new TransformationRecord(
            transformation, mode, originalCode, LocalDateTime.now()
        );
        transformations.put(transformation, record);
        
        addDiagnostic("TRANSFORM_START", 
            String.format("Started transformation: %s (mode: %s)", transformation, mode),
            transformation, DiagnosticLevel.INFO);
    }
    
    /**
     * Record the completion of a transformation.
     */
    public void recordTransformationEnd(String transformation, boolean success, 
                                      String transformedCode, long durationMs, String error) {
        TransformationRecord record = transformations.get(transformation);
        if (record != null) {
            record.endTime = LocalDateTime.now();
            record.success = success;
            record.transformedCode = transformedCode;
            record.durationMs = durationMs;
            record.error = error;
        }
        
        DiagnosticLevel level = success ? DiagnosticLevel.INFO : DiagnosticLevel.ERROR;
        String message = String.format("Transformation %s: %s (took %dms)", 
            transformation, success ? "SUCCESS" : "FAILED", durationMs);
        
        addDiagnostic("TRANSFORM_END", message, transformation, level);
        
        if (error != null) {
            addDiagnostic("TRANSFORM_ERROR", 
                String.format("Error in %s: %s", transformation, error),
                transformation, DiagnosticLevel.ERROR);
        }
    }
    
    /**
     * Record a transformation decision.
     */
    public void recordDecision(String transformation, String reason, boolean applied) {
        String message = String.format("Decision: %s - %s (%s)", 
            transformation, applied ? "APPLIED" : "SKIPPED", reason);
        
        addDiagnostic("TRANSFORM_DECISION", message, transformation, DiagnosticLevel.DEBUG);
    }
    
    /**
     * Record a compatibility check result.
     */
    public void recordCompatibilityCheck(String transformation, String reason, boolean compatible) {
        String message = String.format("Compatibility: %s - %s (%s)", 
            transformation, compatible ? "COMPATIBLE" : "INCOMPATIBLE", reason);
        
        addDiagnostic("COMPATIBILITY_CHECK", message, transformation, DiagnosticLevel.DEBUG);
    }
    
    /**
     * Record a validation failure.
     */
    public void recordValidationFailure(String transformation, String reason) {
        addDiagnostic("VALIDATION_FAILURE", 
            String.format("Validation failed for %s: %s", transformation, reason),
            transformation, DiagnosticLevel.WARN);
    }
    
    /**
     * Record performance metrics.
     */
    public void recordPerformanceMetric(String metric, long value) {
        performance.recordMetric(metric, value);
    }
    
    /**
     * Generate a comprehensive diagnostic report.
     */
    public DiagnosticReport generateReport() {
        return new DiagnosticReport(this);
    }
    
    /**
     * Get all diagnostics for a specific transformation.
     */
    public List<DiagnosticEntry> getDiagnosticsForTransformation(String transformation) {
        return diagnostics.stream()
            .filter(d -> transformation.equals(d.transformation))
            .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
    
    /**
     * Get all diagnostics by level.
     */
    public List<DiagnosticEntry> getDiagnosticsByLevel(DiagnosticLevel level) {
        return diagnostics.stream()
            .filter(d -> d.level == level)
            .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
    
    /**
     * Get transformation records.
     */
    public Map<String, TransformationRecord> getTransformations() {
        return new HashMap<>(transformations);
    }
    
    /**
     * Get performance metrics.
     */
    public PerformanceMetrics getPerformance() {
        return performance;
    }
    
    /**
     * Add a diagnostic entry.
     */
    private void addDiagnostic(String type, String message, String transformation, DiagnosticLevel level) {
        DiagnosticEntry entry = new DiagnosticEntry(
            type, message, transformation, level, LocalDateTime.now()
        );
        diagnostics.add(entry);
    }
    
    /**
     * Generate a unique session ID.
     */
    private String generateSessionId() {
        return "transformation_" + System.currentTimeMillis() + "_" + 
               Integer.toHexString(new Random().nextInt());
    }
    
    /**
     * Diagnostic entry representing a single diagnostic message.
     */
    public static class DiagnosticEntry {
        public final String type;
        public final String message;
        public final String transformation;
        public final DiagnosticLevel level;
        public final LocalDateTime timestamp;
        
        public DiagnosticEntry(String type, String message, String transformation, 
                             DiagnosticLevel level, LocalDateTime timestamp) {
            this.type = type;
            this.message = message;
            this.transformation = transformation;
            this.level = level;
            this.timestamp = timestamp;
        }
        
        @Override
        public String toString() {
            return String.format("[%s] %s - %s: %s", 
                timestamp.format(DateTimeFormatter.ISO_LOCAL_TIME),
                level, transformation, message);
        }
    }
    
    /**
     * Diagnostic levels for categorizing diagnostic messages.
     */
    public enum DiagnosticLevel {
        DEBUG, INFO, WARN, ERROR
    }
    
    /**
     * Record of a transformation execution.
     */
    public static class TransformationRecord {
        public final String transformation;
        public final String mode;
        public final String originalCode;
        public final LocalDateTime startTime;
        
        public LocalDateTime endTime;
        public boolean success;
        public String transformedCode;
        public long durationMs;
        public String error;
        
        public TransformationRecord(String transformation, String mode, String originalCode, LocalDateTime startTime) {
            this.transformation = transformation;
            this.mode = mode;
            this.originalCode = originalCode;
            this.startTime = startTime;
        }
        
        public long getDurationMs() {
            if (endTime != null) {
                return java.time.Duration.between(startTime, endTime).toMillis();
            }
            return 0;
        }
    }
    
    /**
     * Performance metrics collection.
     */
    public static class PerformanceMetrics {
        private final Map<String, List<Long>> metrics;
        
        public PerformanceMetrics() {
            this.metrics = new HashMap<>();
        }
        
        public void recordMetric(String name, long value) {
            metrics.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
        }
        
        public List<Long> getMetric(String name) {
            return metrics.getOrDefault(name, new ArrayList<>());
        }
        
        public double getAverageMetric(String name) {
            List<Long> values = getMetric(name);
            if (values.isEmpty()) return 0.0;
            return values.stream().mapToLong(Long::longValue).average().orElse(0.0);
        }
        
        public long getTotalMetric(String name) {
            return getMetric(name).stream().mapToLong(Long::longValue).sum();
        }
        
        public Map<String, List<Long>> getAllMetrics() {
            return new HashMap<>(metrics);
        }
    }
    
    /**
     * Comprehensive diagnostic report.
     */
    public static class DiagnosticReport {
        private final TransformationDiagnostics diagnostics;
        
        public DiagnosticReport(TransformationDiagnostics diagnostics) {
            this.diagnostics = diagnostics;
        }
        
        /**
         * Generate a summary report.
         */
        public String generateSummary() {
            StringBuilder sb = new StringBuilder();
            sb.append("=== TRANSFORMATION DIAGNOSTIC REPORT ===\n");
            sb.append("Session ID: ").append(diagnostics.sessionId).append("\n");
            sb.append("Start Time: ").append(diagnostics.startTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)).append("\n");
            sb.append("Duration: ").append(getTotalDuration()).append("ms\n\n");
            
            // Transformation summary
            sb.append("=== TRANSFORMATION SUMMARY ===\n");
            Map<String, TransformationRecord> transformations = diagnostics.getTransformations();
            int total = transformations.size();
            int successful = (int) transformations.values().stream().filter(t -> t.success).count();
            int failed = total - successful;
            
            sb.append("Total Transformations: ").append(total).append("\n");
            sb.append("Successful: ").append(successful).append("\n");
            sb.append("Failed: ").append(failed).append("\n");
            sb.append("Success Rate: ").append(total > 0 ? String.format("%.1f%%", (successful * 100.0 / total)) : "0%").append("\n\n");
            
            // Performance summary
            sb.append("=== PERFORMANCE SUMMARY ===\n");
            PerformanceMetrics perf = diagnostics.getPerformance();
            for (String metric : perf.getAllMetrics().keySet()) {
                sb.append(metric).append(": avg=").append(String.format("%.2f", perf.getAverageMetric(metric)))
                  .append("ms, total=").append(perf.getTotalMetric(metric)).append("ms\n");
            }
            sb.append("\n");
            
            // Error summary
            List<DiagnosticEntry> errors = diagnostics.getDiagnosticsByLevel(DiagnosticLevel.ERROR);
            if (!errors.isEmpty()) {
                sb.append("=== ERROR SUMMARY ===\n");
                for (DiagnosticEntry error : errors) {
                    sb.append(error.toString()).append("\n");
                }
            }
            
            return sb.toString();
        }
        
        /**
         * Generate a detailed report.
         */
        public String generateDetailedReport() {
            StringBuilder sb = new StringBuilder();
            sb.append(generateSummary());
            sb.append("\n=== DETAILED TRANSFORMATION RECORDS ===\n");
            
            for (TransformationRecord record : diagnostics.getTransformations().values()) {
                sb.append("Transformation: ").append(record.transformation).append("\n");
                sb.append("Mode: ").append(record.mode).append("\n");
                sb.append("Success: ").append(record.success).append("\n");
                sb.append("Duration: ").append(record.durationMs).append("ms\n");
                if (record.error != null) {
                    sb.append("Error: ").append(record.error).append("\n");
                }
                sb.append("Original Code Length: ").append(record.originalCode != null ? record.originalCode.length() : 0).append("\n");
                sb.append("Transformed Code Length: ").append(record.transformedCode != null ? record.transformedCode.length() : 0).append("\n");
                sb.append("---\n");
            }
            
            return sb.toString();
        }
        
        private long getTotalDuration() {
            if (diagnostics.transformations.isEmpty()) return 0;
            
            LocalDateTime latest = diagnostics.transformations.values().stream()
                .filter(t -> t.endTime != null)
                .map(t -> t.endTime)
                .max(LocalDateTime::compareTo)
                .orElse(diagnostics.startTime);
            
            return java.time.Duration.between(diagnostics.startTime, latest).toMillis();
        }
    }
}
