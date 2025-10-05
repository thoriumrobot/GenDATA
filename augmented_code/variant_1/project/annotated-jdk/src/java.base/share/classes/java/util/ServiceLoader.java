/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.BufferedReader;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.InputStreamReader;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.lang.reflect.Modifier;
    @Positive
import java.net.URL;
    @Positive
import java.net.URLConnection;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Supplier;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;
    @Positive
import sun.nio.cs.UTF_8;
    @Positive
import jdk.internal.loader.BootLoader;
    @Positive
import jdk.internal.loader.ClassLoaders;
    @Positive
import jdk.internal.access.JavaLangAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.misc.VM;
    @Positive
import jdk.internal.module.ServicesCatalog;
    @Positive
import jdk.internal.module.ServicesCatalog.ServiceProvider;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;

    @Positive
@AnnotatedFor({ "interning", "lock", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public final class ServiceLoader<S> implements Iterable<S> {

    @Positive
    public static interface Provider<S> extends Supplier<S> {

    @Positive
        Class<? extends S> type();

    @Positive
        @Override
    @Positive
        S get();
    @Positive
    }

    @Positive
    private static class ProviderImpl<S> implements Provider<S> {

    @Positive
        @Override
    @Positive
        public Class<? extends S> type();

    @Positive
        @Override
    @Positive
        public S get();

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object ob);
    @Positive
    }

    @Positive
    private final class LayerLookupIterator<T> implements Iterator<Provider<T>> {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @Override
    @Positive
        @SideEffectsOnly("this")
    @Positive
        public Provider<T> next(@NonEmpty LayerLookupIterator<T> this);
    @Positive
    }

    @Positive
    private final class ModuleServicesLookupIterator<T> implements Iterator<Provider<T>> {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @Override
    @Positive
        public Provider<T> next(@NonEmpty ModuleServicesLookupIterator<T> this);
    @Positive
    }

    @Positive
    private final class LazyClassPathLookupIterator<T> implements Iterator<Provider<T>> {

    @Positive
        @SuppressWarnings("removal")
    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SuppressWarnings("removal")
    @Positive
        @Override
    @Positive
        @SideEffectsOnly("this")
    @Positive
        public Provider<T> next(@NonEmpty LazyClassPathLookupIterator<T> this);
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Iterator<S> iterator();

    @Positive
    public Stream<Provider<S>> stream();

    @Positive
    private class ProviderSpliterator<T> implements Spliterator<Provider<T>> {

    @Positive
        @Override
    @Positive
        public Spliterator<Provider<T>> trySplit();

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public boolean tryAdvance(Consumer<? super Provider<T>> action);

    @Positive
        @Override
    @Positive
        public int characteristics();

    @Positive
        @Override
    @Positive
        public long estimateSize();
    @Positive
    }

    @Positive
    static <S> ServiceLoader<S> load(Class<S> service, ClassLoader loader, Module callerModule);

    @Positive
    @CallerSensitive
    @Positive
    public static <S> ServiceLoader<S> load(Class<S> service, @Nullable ClassLoader loader);

    @Positive
    @CallerSensitive
    @Positive
    public static <S> ServiceLoader<S> load(Class<S> service);

    @Positive
    @CallerSensitive
    @Positive
    public static <S> ServiceLoader<S> loadInstalled(Class<S> service);

    @Positive
    @CallerSensitive
    @Positive
    public static <S> ServiceLoader<S> load(ModuleLayer layer, Class<S> service);

    @Positive
    public Optional<S> findFirst();

    @Positive
    public void reload();

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied ServiceLoader<S> this);
    @Positive
}
