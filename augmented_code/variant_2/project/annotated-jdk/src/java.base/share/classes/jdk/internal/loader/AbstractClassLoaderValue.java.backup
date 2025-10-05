/*
 * Copyright (c) 2016, 2019, Oracle and/or its affiliates. All rights reserved.
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
package jdk.internal.loader;

import org.checkerframework.checker.nullness.qual.PolyNull;
import jdk.internal.access.JavaLangAccess;
import jdk.internal.access.SharedSecrets;
import java.lang.reflect.UndeclaredThrowableException;
import java.util.Iterator;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;
import java.util.function.Supplier;

public abstract class AbstractClassLoaderValue<CLV extends AbstractClassLoaderValue<CLV, V>, V> {

    public abstract Object key();

    public <K> Sub<K> sub(K key);

    public abstract boolean isEqualOrDescendantOf(AbstractClassLoaderValue<?, V> clv);

    public V get(ClassLoader cl);

    public V putIfAbsent(ClassLoader cl, V v);

    public boolean remove(ClassLoader cl, Object v);

    @PolyNull
    public V computeIfAbsent(ClassLoader cl, BiFunction<? super ClassLoader, ? super CLV, ? extends @PolyNull V> mappingFunction) throws IllegalStateException;

    public void removeAll(ClassLoader cl);

    private static final class Memoizer<CLV extends AbstractClassLoaderValue<CLV, V>, V> implements Supplier<V> {

        @Override
        public V get() throws RecursiveInvocationException;

        static class RecursiveInvocationException extends IllegalStateException {
        }
    }

    public final class Sub<K> extends AbstractClassLoaderValue<Sub<K>, V> {

        public AbstractClassLoaderValue<CLV, V> parent();

        @Override
        public K key();

        @Override
        public boolean isEqualOrDescendantOf(AbstractClassLoaderValue<?, V> clv);

        @Override
        public boolean equals(Object o);

        @Override
        public int hashCode();
    }
}
