/*
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.lang;

import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.mustcall.qual.MustCall;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;
import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.dataflow.qual.TerminatesExecution;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.Console;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.lang.annotation.Annotation;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodType;
import java.lang.invoke.StringConcatFactory;
import java.lang.module.ModuleDescriptor;
import java.lang.reflect.Constructor;
import java.lang.reflect.Executable;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.net.URI;
import java.nio.charset.CharacterCodingException;
import java.security.AccessControlContext;
import java.security.ProtectionDomain;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.nio.channels.Channel;
import java.nio.channels.spi.SelectorProvider;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.PropertyPermission;
import java.util.ResourceBundle;
import java.util.Set;
import java.util.function.Supplier;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;
import jdk.internal.misc.Unsafe;
import jdk.internal.util.StaticProperty;
import jdk.internal.module.ModuleBootstrap;
import jdk.internal.module.ServicesCatalog;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;
import jdk.internal.access.JavaLangAccess;
import jdk.internal.access.SharedSecrets;
import jdk.internal.misc.VM;
import jdk.internal.logger.LoggerFinderLoader;
import jdk.internal.logger.LazyLoggers;
import jdk.internal.logger.LocalizedLoggerWrapper;
import jdk.internal.util.SystemProps;
import jdk.internal.vm.annotation.IntrinsicCandidate;
import jdk.internal.vm.annotation.Stable;
import sun.nio.fs.DefaultFileSystemProvider;
import sun.reflect.annotation.AnnotationType;
import sun.nio.ch.Interruptible;
import sun.security.util.SecurityConstants;

@AnnotatedFor({ "index", "interning", "lock", "mustcall", "nullness", "signedness" })
@UsesObjectEquals
public final class System {

    @CFComment("This field can be null. The Checker Framework conservatively annotates it as @NonNull, forbidding programs that set it to null.")
    @MustCall({})
    public static final InputStream in;

    @CFComment("This field can be null. The Checker Framework conservatively annotates it as @NonNull, forbidding programs that set it to null.")
    @MustCall({})
    public static final PrintStream out;

    @CFComment("This field can be null. The Checker Framework conservatively annotates it as @NonNull, forbidding programs that set it to null.")
    @MustCall({})
    public static final PrintStream err;

    @CFComment("Null is a legal argument. The Checker Framework conservatively forbids programs that pass null.")
    public static void setIn(InputStream in);

    @CFComment("Null is a legal argument. The Checker Framework conservatively forbids programs that pass null.")
    public static void setOut(PrintStream out);

    @CFComment("Null is a legal argument. The Checker Framework conservatively forbids programs that pass null.")
    public static void setErr(PrintStream err);

    @Nullable
    public static Console console();

    @Nullable
    public static Channel inheritedChannel() throws IOException;

    @Deprecated()
    public static void setSecurityManager(@SuppressWarnings("removal") @Nullable SecurityManager sm);

    @SuppressWarnings("removal")
    @Deprecated()
    @Nullable
    public static SecurityManager getSecurityManager();

    @IntrinsicCandidate
    public static native long currentTimeMillis();

    @IntrinsicCandidate
    public static native long nanoTime();

    @SideEffectFree
    @IntrinsicCandidate
    public static native void arraycopy(@PolySigned @GuardSatisfied Object src, @NonNegative int srcPos, @PolySigned @GuardSatisfied Object dest, @NonNegative int destPos, @LTLengthOf(value = { "#1", "#3" }, offset = { "#2 - 1", "#4 - 1" }) @NonNegative int length);

    @Pure
    @IntrinsicCandidate
    public static native int identityHashCode(@GuardSatisfied @Nullable Object x);

    public static Properties getProperties();

    @Pure
    public static String lineSeparator();

    public static void setProperties(@Nullable Properties props);

    @Pure
    @Nullable
    public static String getProperty(String key);

    @Pure
    @PolyNull
    public static String getProperty(String key, @PolyNull String def);

    @Nullable
    public static String setProperty(String key, String value);

    @Nullable
    public static String clearProperty(String key);

    @Nullable
    public static String getenv(String name);

    public static java.util.Map<String, String> getenv();

    public interface Logger {

        public enum Level {

            ALL(Integer.MIN_VALUE),
            TRACE(400),
            DEBUG(500),
            INFO(800),
            WARNING(900),
            ERROR(1000),
            OFF(Integer.MAX_VALUE);

            private final int severity;

            private Level(int severity) {
            }

            public final String getName();

            public final int getSeverity();
        }

        public String getName();

        public boolean isLoggable(Level level);

        public default void log(Level level, String msg);

        public default void log(Level level, Supplier<String> msgSupplier);

        public default void log(Level level, Object obj);

        public default void log(Level level, String msg, Throwable thrown);

        public default void log(Level level, Supplier<String> msgSupplier, Throwable thrown);

        public default void log(Level level, String format, Object... params);

        public void log(Level level, ResourceBundle bundle, String msg, Throwable thrown);

        public void log(Level level, ResourceBundle bundle, String format, Object... params);
    }

    public static abstract class LoggerFinder {

        protected LoggerFinder() {
        }

        public abstract Logger getLogger(String name, Module module);

        public Logger getLocalizedLogger(String name, ResourceBundle bundle, Module module);

        public static LoggerFinder getLoggerFinder();

        @SuppressWarnings("removal")
        static LoggerFinder accessProvider();
    }

    @CallerSensitive
    public static Logger getLogger(String name);

    @SuppressWarnings("removal")
    @CallerSensitive
    public static Logger getLogger(String name, ResourceBundle bundle);

    @TerminatesExecution
    public static void exit(int status);

    public static void gc();

    public static void runFinalization();

    @CallerSensitive
    public static void load(String filename);

    @CallerSensitive
    public static void loadLibrary(String libname);

    public static native String mapLibraryName(String libname);
}
