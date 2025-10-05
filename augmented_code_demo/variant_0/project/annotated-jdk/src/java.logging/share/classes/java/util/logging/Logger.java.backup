/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util.logging;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signature.qual.BinaryName;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.lang.ref.WeakReference;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Locale;
import java.util.MissingResourceException;
import java.util.Objects;
import java.util.ResourceBundle;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Supplier;
import jdk.internal.access.JavaUtilResourceBundleAccess;
import jdk.internal.access.SharedSecrets;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;
import static jdk.internal.logger.DefaultLoggerFinder.isSystem;

@CFComment({ "lock: inherited methods", "public boolean isEmpty(@GuardSatisfied LinkedList<E> this) { throw new RuntimeException(\"skeleton method\"); }", "public boolean containsAll(@GuardSatisfied LinkedList<E> this, Collection<?> c);", "public int hashCode(@GuardSatisfied LinkedList<E> this);", "public boolean equals(@GuardSatisfied LinkedList<E> this, Object o);" })
@AnnotatedFor({ "index", "interning", "lock", "signature" })
@UsesObjectEquals
public class Logger {

    private static final class LoggerBundle {

        boolean isSystemBundle();

        static LoggerBundle get(String name, ResourceBundle bundle);
    }

    private static final class RbAccess {
    }

    private static final class ConfigurationData {

        void setUseParentHandlers(boolean flag);

        void setFilter(Filter f);

        void setLevelObject(Level l);

        void setLevelValue(int v);

        void addHandler(Handler h);

        void removeHandler(Handler h);

        ConfigurationData merge(Logger systemPeer);
    }

    @Interned
    public static final String GLOBAL_LOGGER_NAME;

    @Pure
    public static final Logger getGlobal();

    @Deprecated
    public static final Logger global;

    protected Logger(@Nullable String name, @Nullable @BinaryName String resourceBundleName) {
    }

    final void mergeWithSystemLogger(Logger system);

    void setLogManager(@GuardSatisfied Logger this, LogManager manager);

    private static class SystemLoggerHelper {
    }

    @Pure
    @CallerSensitive
    public static Logger getLogger(String name);

    @Pure
    @CallerSensitive
    public static Logger getLogger(String name, @Nullable @BinaryName String resourceBundleName);

    static Logger getPlatformLogger(String name);

    @Pure
    public static Logger getAnonymousLogger();

    @Pure
    @CallerSensitive
    public static Logger getAnonymousLogger(@Nullable @BinaryName String resourceBundleName);

    @Pure
    @Nullable
    public ResourceBundle getResourceBundle(@GuardSatisfied Logger this);

    @Pure
    @Nullable
    @BinaryName
    public String getResourceBundleName(@GuardSatisfied Logger this);

    public void setFilter(@GuardSatisfied Logger this, @Nullable Filter newFilter) throws SecurityException;

    @Pure
    @Nullable
    public Filter getFilter(@GuardSatisfied Logger this);

    @SideEffectFree
    public void log(@GuardSatisfied Logger this, LogRecord record);

    @SideEffectFree
    public void log(@GuardSatisfied Logger this, @GuardSatisfied Level level, @Nullable String msg);

    @SideEffectFree
    public void log(@GuardSatisfied Logger this, @GuardSatisfied Level level, @GuardSatisfied Supplier<String> msgSupplier);

    @SideEffectFree
    public void log(@GuardSatisfied Logger this, @GuardSatisfied Level level, @Nullable String msg, @GuardSatisfied @Nullable Object param1);

    @SideEffectFree
    public void log(@GuardSatisfied Logger this, @GuardSatisfied Level level, @Nullable String msg, @Nullable Object @GuardSatisfied @Nullable [] params);

    @SideEffectFree
    public void log(@GuardSatisfied Logger this, @GuardSatisfied Level level, @Nullable String msg, @GuardSatisfied @Nullable Throwable thrown);

    @SideEffectFree
    public void log(@GuardSatisfied Logger this, @GuardSatisfied Level level, @GuardSatisfied @Nullable Throwable thrown, @GuardSatisfied Supplier<String> msgSupplier);

    @SideEffectFree
    public void logp(@GuardSatisfied Logger this, @GuardSatisfied Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable String msg);

    @SideEffectFree
    public void logp(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, Supplier<String> msgSupplier);

    @SideEffectFree
    public void logp(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable String msg, @Nullable Object param1);

    @SideEffectFree
    public void logp(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable String msg, @Nullable Object @Nullable [] params);

    @SideEffectFree
    public void logp(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable String msg, @Nullable Throwable thrown);

    @SideEffectFree
    public void logp(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable Throwable thrown, Supplier<String> msgSupplier);

    @SideEffectFree
    @Deprecated
    public void logrb(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable @BinaryName String bundleName, @Nullable String msg);

    @SideEffectFree
    @Deprecated
    public void logrb(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable @BinaryName String bundleName, @Nullable String msg, @Nullable Object param1);

    @SideEffectFree
    @Deprecated
    public void logrb(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable @BinaryName String bundleName, @Nullable String msg, @Nullable Object @Nullable [] params);

    public void logrb(Level level, String sourceClass, String sourceMethod, ResourceBundle bundle, String msg, Object... params);

    public void logrb(Level level, ResourceBundle bundle, String msg, Object... params);

    @SideEffectFree
    @Deprecated
    public void logrb(@GuardSatisfied Logger this, Level level, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable @BinaryName String bundleName, @Nullable String msg, @Nullable Throwable thrown);

    public void logrb(Level level, String sourceClass, String sourceMethod, ResourceBundle bundle, String msg, Throwable thrown);

    public void logrb(Level level, ResourceBundle bundle, String msg, Throwable thrown);

    @SideEffectFree
    public void entering(@GuardSatisfied Logger this, @Nullable String sourceClass, @Nullable String sourceMethod);

    @SideEffectFree
    public void entering(@GuardSatisfied Logger this, @Nullable String sourceClass, @Nullable String sourceMethod, @GuardSatisfied @Nullable Object param1);

    @SideEffectFree
    public void entering(@GuardSatisfied Logger this, @Nullable String sourceClass, @Nullable String sourceMethod, @Nullable Object @GuardSatisfied @Nullable [] params);

    @SideEffectFree
    public void exiting(@GuardSatisfied Logger this, @Nullable String sourceClass, @Nullable String sourceMethod);

    @SideEffectFree
    public void exiting(@GuardSatisfied Logger this, @Nullable String sourceClass, @Nullable String sourceMethod, @GuardSatisfied @Nullable Object result);

    @SideEffectFree
    public void throwing(@GuardSatisfied Logger this, @Nullable String sourceClass, @Nullable String sourceMethod, @GuardSatisfied @Nullable Throwable thrown);

    @SideEffectFree
    public void severe(@GuardSatisfied Logger this, @Nullable String msg);

    @SideEffectFree
    public void warning(@GuardSatisfied Logger this, @Nullable String msg);

    @SideEffectFree
    public void info(@GuardSatisfied Logger this, @Nullable String msg);

    @SideEffectFree
    public void config(@GuardSatisfied Logger this, @Nullable String msg);

    @SideEffectFree
    public void fine(@GuardSatisfied Logger this, @Nullable String msg);

    @SideEffectFree
    public void finer(@GuardSatisfied Logger this, @Nullable String msg);

    @SideEffectFree
    public void finest(@GuardSatisfied Logger this, @Nullable String msg);

    @SideEffectFree
    public void severe(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @SideEffectFree
    public void warning(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @SideEffectFree
    public void info(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @SideEffectFree
    public void config(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @SideEffectFree
    public void fine(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @SideEffectFree
    public void finer(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    @SideEffectFree
    public void finest(@GuardSatisfied Logger this, Supplier<String> msgSupplier);

    public void setLevel(@GuardSatisfied Logger this, @Nullable Level newLevel) throws SecurityException;

    final boolean isLevelInitialized();

    @Pure
    @Nullable
    public Level getLevel(@GuardSatisfied Logger this);

    @Pure
    public boolean isLoggable(Level level);

    @Pure
    @Nullable
    public String getName(@GuardSatisfied Logger this);

    public void addHandler(@GuardSatisfied Logger this, Handler handler) throws SecurityException;

    public void removeHandler(@GuardSatisfied Logger this, @Nullable Handler handler) throws SecurityException;

    @SideEffectFree
    public Handler[] getHandlers(@GuardSatisfied Logger this);

    Handler[] accessCheckedHandlers();

    public void setUseParentHandlers(@GuardSatisfied Logger this, boolean useParentHandlers);

    @Pure
    public boolean getUseParentHandlers(@GuardSatisfied Logger this);

    public void setResourceBundle(ResourceBundle bundle);

    @Pure
    @Nullable
    public Logger getParent(@GuardSatisfied Logger this);

    public void setParent(@GuardSatisfied Logger this, @GuardSatisfied Logger parent);

    final void removeChildLogger(@GuardSatisfied Logger this, LogManager.LoggerWeakRef child);
}
