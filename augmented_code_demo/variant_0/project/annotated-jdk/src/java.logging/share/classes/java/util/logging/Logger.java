/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util.logging;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signature.qual.BinaryName;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Locale;
    @Positive
import java.util.MissingResourceException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.concurrent.CopyOnWriteArrayList;
    @Positive
import java.util.function.Supplier;
    @Positive
import jdk.internal.access.JavaUtilResourceBundleAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import static jdk.internal.logger.DefaultLoggerFinder.isSystem;

    @Positive
@CFComment({ "lock: inherited methods", "public boolean isEmpty(@GuardSatisfied LinkedList<E> this) { throw new RuntimeException(\"skeleton method\"); }", "public boolean containsAll(@GuardSatisfied LinkedList<E> this, Collection<?> c);", "public int hashCode(@GuardSatisfied LinkedList<E> this);", "public boolean equals(@GuardSatisfied LinkedList<E> this, Object o);" })
    @Positive
@AnnotatedFor({ "index", "interning", "lock", "signature" })
    @Positive
@UsesObjectEquals
    @Positive
public class Logger {

    @Positive
    private static final class LoggerBundle {

    @Positive
        boolean isSystemBundle();

    @Positive
        static LoggerBundle get(String name, ResourceBundle bundle);
    @Positive
    }

    @Positive
    private static final class RbAccess {
    @Positive
    }

    @Positive
    private static final class ConfigurationData {

    @Positive
        void setUseParentHandlers(boolean flag);

    @Positive
        void setFilter(Filter f);

    @Positive
        void setLevelObject(Level l);

    @Positive
        void setLevelValue(int v);

    @Positive
        void addHandler(Handler h);

    @Positive
        void removeHandler(Handler h);

    @Positive
        ConfigurationData merge(Logger systemPeer);
    @Positive
    }

    @Positive
    @Interned
    @Positive
    public static final String GLOBAL_LOGGER_NAME;

    @Positive
    @Pure
    @Positive
    public static final Logger getGlobal();

    @Positive
    @Deprecated
    @Positive
    public static final Logger global;

    @Positive
    protected Logger(@Nullable String name, @Nullable @BinaryName String resourceBundleName) {
    @Positive
    }

    @Positive
    final void mergeWithSystemLogger(Logger system);

    @Positive
    void setLogManager(@GuardSatisfied Logger this, LogManager manager);

    @Positive
    private static class SystemLoggerHelper {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    public static Logger getLogger(String name);

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    public static Logger getLogger(String name, @Nullable @BinaryName String resourceBundleName);

    @Positive
    static Logger getPlatformLogger(String name);

    @Positive
    @Pure
    @Positive
    public static Logger getAnonymousLogger();

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    public static Logger getAnonymousLogger(@Nullable @BinaryName String resourceBundleName);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public ResourceBundle getResourceBundle(@GuardSatisfied Logger this);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    @BinaryName
    @Positive
    public String getResourceBundleName(@GuardSatisfied Logger this);

    @Positive
    public void setFilter(@GuardSatisfied Logger this, @Nullable Filter newFilter) throws SecurityException;

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Filter getFilter(@GuardSatisfied Logger this);

    @Positive
    @SideEffectFree
    @Positive
    public void log(@GuardSatisfied Logger this, LogRecord record);

    @Positive
    @SideEffectFree
    @Positive
    public void log(@GuardSatisfied Logger this, @GuardSatisfied Level level, @Nullable String msg);

    @Positive
    @SideEffectFree
    @Positive
    public void log(@GuardSatisfied Logger this, @GuardSatisfied Level level, @GuardSatisfied Supplier<String> msgSupplier);

    @Positive
    @SideEffectFree
    @Positive
    public void log(@GuardSatisfied Logger this, @GuardSatisfied Level level, @Nullable String msg, @GuardSatisfied @Nullable Object param1);

    @Positive
    @SideEffectFree
    @Positive
    public void log(@GuardSatisfied Logger this, @GuardSatisfied Level level, @Nullable String msg, @Nullable Object @GuardSatisfied @Nullable [] params);

    @Positive
    @SideEffectFree
    @Positive
    public void log(@GuardSatisfied Logger this, @GuardSatisfied Level level, @Nullable String msg, @GuardSatisfied @Nullable Throwable thrown);

    @Positive
    @SideEffectFree
    @Positive
    public void log(@GuardSatisfied Logger this, @GuardSatisfied Level level, @GuardSatisfied @Nullable Throwable thrown, @GuardSatisfied Supplier<String> msgSupplier);

    @Positive
    @SideEffectFree
    @Positive
    public void logp(@GuardSatisfied Logger this, @GuardSatisfied Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable String msg);

    @Positive
    @SideEffectFree
    @Positive
    public void logp(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, Supplier<String> msgSupplier);

    @Positive
    @SideEffectFree
    @Positive
    public void logp(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable String msg, @Nullable Object param1);

    @Positive
    @SideEffectFree
    @Positive
    public void logp(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable String msg, @Nullable Object @Nullable [] params);

    @Positive
    @SideEffectFree
    @Positive
    public void logp(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable String msg, @Nullable Throwable thrown);

    @Positive
    @SideEffectFree
    @Positive
    public void logp(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable Throwable thrown, Supplier<String> msgSupplier);

    @Positive
    @SideEffectFree
    @Positive
    @Deprecated
    @Positive
    public void logrb(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable @BinaryName String bundleName, @Nullable String msg);

    @Positive
    @SideEffectFree
    @Positive
    @Deprecated
    @Positive
    public void logrb(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable @BinaryName String bundleName, @Nullable String msg, @Nullable Object param1);

    @Positive
    @SideEffectFree
    @Positive
    @Deprecated
    @Positive
    public void logrb(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable @BinaryName String bundleName, @Nullable String msg, @Nullable Object @Nullable [] params);

    @Positive
    public void logrb(Level level, String sourceClass, String sourceMethod, ResourceBundle bundle, String msg, Object... params);

    @Positive
    public void logrb(Level level, ResourceBundle bundle, String msg, Object... params);

    @Positive
    @SideEffectFree
    @Positive
    @Deprecated
    @Positive
    public void logrb(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable @BinaryName String bundleName, @Nullable String msg, @Nullable Throwable thrown);

    @Positive
    public void logrb(Level level, String sourceClass, String sourceMethod, ResourceBundle bundle, String msg, Throwable thrown);

    @Positive
    public void logrb(Level level, ResourceBundle bundle, String msg, Throwable thrown);

    @Positive
    @SideEffectFree
    @Positive
    public void entering(@GuardSatisfied Logger this, @Nullable String sourceClass, @Nullable String sourceMethod);

    @Positive
    @SideEffectFree
    @Positive
    public void entering(@GuardSatisfied Logger this, @Nullable String sourceClass, @Nullable String sourceMethod, @GuardSatisfied @Nullable Object param1);

    @Positive
    @SideEffectFree
    @Positive
    public void entering(@GuardSatisfied Logger this, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable Object @GuardSatisfied @Nullable [] params);

    @Positive
    @SideEffectFree
    @Positive
    public void exiting(@GuardSatisfied Logger this, @Nullable String sourceClass, @Nullable String sourceMethod);

    @Positive
    @SideEffectFree
    @Positive
    public void exiting(@GuardSatisfied Logger this, @Nullable String sourceClass, @Nullable String sourceMethod, @GuardSatisfied @Nullable Object result);

    @Positive
    @SideEffectFree
    @Positive
    public void throwing(@GuardSatisfied Logger this, @Nullable String sourceClass, @Nullable String sourceMethod, @GuardSatisfied @Nullable Throwable thrown);

    @Positive
    @SideEffectFree
    @Positive
    public void severe(@GuardSatisfied Logger this, @Nullable String msg);

    @Positive
    @SideEffectFree
    @Positive
    public void warning(@GuardSatisfied Logger this, @Nullable String msg);

    @Positive
    @SideEffectFree
    @Positive
    public void info(@GuardSatisfied Logger this, @Nullable String msg);

    @Positive
    @SideEffectFree
    @Positive
    public void config(@GuardSatisfied Logger this, @Nullable String msg);

    @Positive
    @SideEffectFree
    @Positive
    public void fine(@GuardSatisfied Logger this, @Nullable String msg);

    @Positive
    @SideEffectFree
    @Positive
    public void finer(@GuardSatisfied Logger this, @Nullable String msg);

    @Positive
    @SideEffectFree
    @Positive
    public void finest(@GuardSatisfied Logger this, @Nullable String msg);

    @Positive
    @SideEffectFree
    @Positive
    public void severe(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @Positive
    @SideEffectFree
    @Positive
    public void warning(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @Positive
    @SideEffectFree
    @Positive
    public void info(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @Positive
    @SideEffectFree
    @Positive
    public void config(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @Positive
    @SideEffectFree
    @Positive
    public void fine(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @Positive
    @SideEffectFree
    @Positive
    public void finer(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @Positive
    @SideEffectFree
    @Positive
    public void finest(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @Positive
    public void setLevel(@GuardSatisfied Logger this, @Nullable Level newLevel) throws SecurityException;

    @Positive
    final boolean isLevelInitialized();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Level getLevel(@GuardSatisfied Logger this);

    @Positive
    @Pure
    @Positive
    public boolean isLoggable(Level level);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getName(@GuardSatisfied Logger this);

    @Positive
    public void addHandler(@GuardSatisfied Logger this, Handler handler) throws SecurityException;

    @Positive
    public void removeHandler(@GuardSatisfied Logger this, @Nullable Handler handler) throws SecurityException;

    @Positive
    @SideEffectFree
    @Positive
    public Handler[] getHandlers(@GuardSatisfied Logger this);

    @Positive
    Handler[] accessCheckedHandlers();

    @Positive
    public void setUseParentHandlers(@GuardSatisfied Logger this, boolean useParentHandlers);

    @Positive
    @Pure
    @Positive
    public boolean getUseParentHandlers(@GuardSatisfied Logger this);

    @Positive
    public void setResourceBundle(ResourceBundle bundle);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Logger getParent(@GuardSatisfied Logger this);

    @Positive
    public void setParent(@GuardSatisfied Logger this, @GuardSatisfied Logger parent);

    @Positive
    final void removeChildLogger(@GuardSatisfied Logger this, LogManager.LoggerWeakRef child);
    @Positive
}

// CFWR semantic augmentation - variant 0
