/*
    @Positive
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.lang;

    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCall;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.TerminatesExecution;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.io.BufferedInputStream;
    @Positive
import java.io.BufferedOutputStream;
    @Positive
import java.io.Console;
    @Positive
import java.io.FileDescriptor;
    @Positive
import java.io.FileInputStream;
    @Positive
import java.io.FileOutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.PrintStream;
    @Positive
import java.io.UnsupportedEncodingException;
    @Positive
import java.lang.annotation.Annotation;
    @Positive
import java.lang.invoke.MethodHandle;
    @Positive
import java.lang.invoke.MethodType;
    @Positive
import java.lang.invoke.StringConcatFactory;
    @Positive
import java.lang.module.ModuleDescriptor;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.lang.reflect.Executable;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.lang.reflect.Modifier;
    @Positive
import java.net.URI;
    @Positive
import java.nio.charset.CharacterCodingException;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.ProtectionDomain;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.nio.channels.Channel;
    @Positive
import java.nio.channels.spi.SelectorProvider;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Properties;
    @Positive
import java.util.PropertyPermission;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.Supplier;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.stream.Stream;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import jdk.internal.util.StaticProperty;
    @Positive
import jdk.internal.module.ModuleBootstrap;
    @Positive
import jdk.internal.module.ServicesCatalog;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.access.JavaLangAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.misc.VM;
    @Positive
import jdk.internal.logger.LoggerFinderLoader;
    @Positive
import jdk.internal.logger.LazyLoggers;
    @Positive
import jdk.internal.logger.LocalizedLoggerWrapper;
    @Positive
import jdk.internal.util.SystemProps;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import jdk.internal.vm.annotation.Stable;
    @Positive
import sun.nio.fs.DefaultFileSystemProvider;
    @Positive
import sun.reflect.annotation.AnnotationType;
    @Positive
import sun.nio.ch.Interruptible;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "mustcall", "nullness", "signedness" })
    @Positive
@UsesObjectEquals
    @Positive
public final class System {

    @Positive
    @CFComment("This field can be null. The Checker Framework conservatively annotates it as @NonNull, forbidding programs that set it to null.")
    @Positive
    @MustCall({})
    @Positive
    public static final InputStream in;

    @Positive
    @CFComment("This field can be null. The Checker Framework conservatively annotates it as @NonNull, forbidding programs that set it to null.")
    @Positive
    @MustCall({})
    @Positive
    public static final PrintStream out;

    @Positive
    @CFComment("This field can be null. The Checker Framework conservatively annotates it as @NonNull, forbidding programs that set it to null.")
    @Positive
    @MustCall({})
    @Positive
    public static final PrintStream err;

    @Positive
    @CFComment("Null is a legal argument. The Checker Framework conservatively forbids programs that pass null.")
    @Positive
    public static void setIn(InputStream in);

    @Positive
    @CFComment("Null is a legal argument. The Checker Framework conservatively forbids programs that pass null.")
    @Positive
    public static void setOut(PrintStream out);

    @Positive
    @CFComment("Null is a legal argument. The Checker Framework conservatively forbids programs that pass null.")
    @Positive
    public static void setErr(PrintStream err);

    @Positive
    @Nullable
    @Positive
    public static Console console();

    @Positive
    @Nullable
    @Positive
    public static Channel inheritedChannel() throws IOException;

    @Positive
    @Deprecated()
    @Positive
    public static void setSecurityManager(@SuppressWarnings("removal") @Nullable SecurityManager sm);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @Deprecated()
    @Positive
    @Nullable
    @Positive
    public static SecurityManager getSecurityManager();

    @Positive
    @IntrinsicCandidate
    @Positive
    public static native long currentTimeMillis();

    @Positive
    @IntrinsicCandidate
    @Positive
    public static native long nanoTime();

    @Positive
    @SideEffectFree
    @Positive
    @IntrinsicCandidate
    @Positive
    public static native void arraycopy(@PolySigned @GuardSatisfied Object src, @NonNegative int srcPos, @PolySigned @GuardSatisfied Object dest, @NonNegative int destPos, @LTLengthOf(value = { "#1", "#3" }, offset = { "#2 - 1", "#4 - 1" }) @NonNegative int length);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static native int identityHashCode(@GuardSatisfied @Nullable Object x);

    @Positive
    public static Properties getProperties();

    @Positive
    @Pure
    @Positive
    public static String lineSeparator();

    @Positive
    public static void setProperties(@Nullable Properties props);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public static String getProperty(String key);

    @Positive
    @Pure
    @Positive
    @PolyNull
    @Positive
    public static String getProperty(String key, @PolyNull String def);

    @Positive
    @Nullable
    @Positive
    public static String setProperty(String key, String value);

    @Positive
    @Nullable
    @Positive
    public static String clearProperty(String key);

    @Positive
    @Nullable
    @Positive
    public static String getenv(String name);

    @Positive
    public static java.util.Map<String, String> getenv();

    @Positive
    public interface Logger {

    @Positive
        public enum Level {

    @Positive
            ALL(Integer.MIN_VALUE),
    @Positive
            TRACE(400),
    @Positive
            DEBUG(500),
    @Positive
            INFO(800),
    @Positive
            WARNING(900),
    @Positive
            ERROR(1000),
    @Positive
            OFF(Integer.MAX_VALUE);

    @Positive
            private final int severity;

    @Positive
            private Level(int severity) {
    @Positive
            }

    @Positive
            public final String getName();

    @Positive
            public final int getSeverity();
    @Positive
        }

    @Positive
        public String getName();

    @Positive
        public boolean isLoggable(Level level);

    @Positive
        public default void log(Level level, String msg);

    @Positive
        public default void log(Level level, Supplier<String> msgSupplier);

    @Positive
        public default void log(Level level, Object obj);

    @Positive
        public default void log(Level level, String msg, Throwable thrown);

    @Positive
        public default void log(Level level, Supplier<String> msgSupplier, Throwable thrown);

    @Positive
        public default void log(Level level, String format, Object... params);

    @Positive
        public void log(Level level, ResourceBundle bundle, String msg, Throwable thrown);

    @Positive
        public void log(Level level, ResourceBundle bundle, String format, Object... params);
    @Positive
    }

    @Positive
    public static abstract class LoggerFinder {

    @Positive
        protected LoggerFinder() {
    @Positive
        }

    @Positive
        public abstract Logger getLogger(String name, Module module);

    @Positive
        public Logger getLocalizedLogger(String name, ResourceBundle bundle, Module module);

    @Positive
        public static LoggerFinder getLoggerFinder();

    @Positive
        @SuppressWarnings("removal")
    @Positive
        static LoggerFinder accessProvider();
    @Positive
    }

    @Positive
    @CallerSensitive
    @Positive
    public static Logger getLogger(String name);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @CallerSensitive
    @Positive
    public static Logger getLogger(String name, ResourceBundle bundle);

    @Positive
    @TerminatesExecution
    @Positive
    public static void exit(int status);

    @Positive
    public static void gc();

    @Positive
    public static void runFinalization();

    @Positive
    @CallerSensitive
    @Positive
    public static void load(String filename);

    @Positive
    @CallerSensitive
    @Positive
    public static void loadLibrary(String libname);

    @Positive
    public static native String mapLibraryName(String libname);
    @Positive
}

// CFWR semantic augmentation - variant 1
