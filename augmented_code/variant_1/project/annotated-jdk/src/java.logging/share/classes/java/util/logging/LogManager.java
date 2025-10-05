/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.util.logging;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.signature.qual.BinaryName;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.util.*;
    @Positive
import java.security.*;
    @Positive
import java.lang.ref.ReferenceQueue;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.nio.file.Paths;
    @Positive
import java.util.concurrent.CopyOnWriteArrayList;
    @Positive
import java.util.concurrent.locks.ReentrantLock;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.stream.Collectors;
    @Positive
import java.util.stream.Stream;
    @Positive
import jdk.internal.access.JavaAWTAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import sun.util.logging.internal.LoggingProviderImpl;
    @Positive
import static jdk.internal.logger.DefaultLoggerFinder.isSystem;

    @Positive
@AnnotatedFor({ "index", "interning", "signature" })
    @Positive
@UsesObjectEquals
    @Positive
public class LogManager {

    @Positive
    private static final class CloseOnReset {

    @Positive
        @Override
    @Positive
        public boolean equals(Object other);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        public Logger get();

    @Positive
        public static CloseOnReset create(Logger logger);
    @Positive
    }

    @Positive
    private class Cleaner extends Thread {

    @Positive
        @Override
    @Positive
        public void run();
    @Positive
    }

    @Positive
    protected LogManager() {
    @Positive
    }

    @Positive
    @SuppressWarnings("removal")
    @Positive
    final void ensureLogManagerInitialized();

    @Positive
    public static LogManager getLogManager();

    @Positive
    final LoggerContext getSystemContext();

    @Positive
    Logger demandLogger(String name, String resourceBundleName, Class<?> caller);

    @Positive
    Logger demandLogger(String name, String resourceBundleName, Module module);

    @Positive
    Logger demandSystemLogger(String name, String resourceBundleName, Class<?> caller);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    Logger demandSystemLogger(String name, String resourceBundleName, Module module);

    @Positive
    class LoggerContext {

    @Positive
        final boolean requiresDefaultLoggers();

    @Positive
        final LogManager getOwner();

    @Positive
        final Logger getRootLogger();

    @Positive
        final Logger getGlobalLogger();

    @Positive
        Logger demandLogger(String name, String resourceBundleName, Module module);

    @Positive
        Logger findLogger(String name);

    @Positive
        boolean addLocalLogger(Logger logger);

    @Positive
        synchronized boolean addLocalLogger(Logger logger, boolean addDefaultLoggersIfNeeded);

    @Positive
        void removeLoggerRef(String name, LoggerWeakRef ref);

    @Positive
        synchronized Enumeration<String> getLoggerNames();

    @Positive
        LogNode getNode(String name);
    @Positive
    }

    @Positive
    final class SystemLoggerContext extends LoggerContext {

    @Positive
        @Override
    @Positive
        Logger demandLogger(String name, String resourceBundleName, Module module);
    @Positive
    }

    @Positive
    final class LoggerWeakRef extends WeakReference<Logger> {

    @Positive
        void dispose();

    @Positive
        void setNode(LogNode node);

    @Positive
        void setParentRef(WeakReference<Logger> parentRef);
    @Positive
    }

    @Positive
    final void drainLoggerRefQueueBounded();

    @Positive
    public boolean addLogger(Logger logger);

    @Positive
    public Logger getLogger(String name);

    @Positive
    public Enumeration<String> getLoggerNames();

    @Positive
    public void readConfiguration() throws IOException, SecurityException;

    @Positive
    String getConfigurationFileName() throws IOException;

    @Positive
    public void reset() throws SecurityException;

    @Positive
    public void readConfiguration(InputStream ins) throws IOException, SecurityException;

    @Positive
    static final class VisitedLoggers implements Predicate<Logger> {

    @Positive
        @Override
    @Positive
        public boolean test(Logger logger);

    @Positive
        public void clear();
    @Positive
    }

    @Positive
    public void updateConfiguration(Function<String, BiFunction<String, String, String>> mapper) throws IOException;

    @Positive
    public void updateConfiguration(InputStream ins, Function<String, BiFunction<String, String, String>> mapper) throws IOException;

    @Positive
    public String getProperty(String name);

    @Positive
    String getStringProperty(String name, String defaultValue);

    @Positive
    int getIntProperty(String name, int defaultValue);

    @Positive
    long getLongProperty(String name, long defaultValue);

    @Positive
    boolean getBooleanProperty(String name, boolean defaultValue);

    @Positive
    Level getLevelProperty(String name, Level defaultValue);

    @Positive
    @SuppressWarnings("signature")
    @Positive
    Filter getFilterProperty(String name, Filter defaultValue);

    @Positive
    Formatter getFormatterProperty(String name, Formatter defaultValue);

    @Positive
    void checkPermission();

    @Positive
    @Deprecated()
    @Positive
    public void checkAccess() throws SecurityException;

    @Positive
    private static class LogNode {

    @Positive
        void walkAndSetParent(Logger parent);
    @Positive
    }

    @Positive
    private final class RootLogger extends Logger {

    @Positive
        @Override
    @Positive
        public void log(LogRecord record);

    @Positive
        @Override
    @Positive
        public void addHandler(Handler h);

    @Positive
        @Override
    @Positive
        public void removeHandler(Handler h);

    @Positive
        @Override
    @Positive
        Handler[] accessCheckedHandlers();
    @Positive
    }

    @Positive
    public static final String LOGGING_MXBEAN_NAME;

    @Positive
    @Deprecated()
    @Positive
    public static synchronized LoggingMXBean getLoggingMXBean();

    @Positive
    public LogManager addConfigurationListener(Runnable listener);

    @Positive
    public void removeConfigurationListener(Runnable listener);

    @Positive
    private static final class LoggingProviderAccess implements LoggingProviderImpl.LogManagerAccess, PrivilegedAction<Void> {

    @Positive
        @Override
    @Positive
        public Logger demandLoggerFor(LogManager manager, String name, Module module);

    @Positive
        @Override
    @Positive
        public Void run();
    @Positive
    }
    @Positive
}
