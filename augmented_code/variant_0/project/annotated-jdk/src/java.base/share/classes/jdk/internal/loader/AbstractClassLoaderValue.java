/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2016, 2019, Oracle and/or its affiliates. All rights reserved.
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
package jdk.internal.loader;

    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import jdk.internal.access.JavaLangAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import java.lang.reflect.UndeclaredThrowableException;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Objects;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.Supplier;

    @Positive
public abstract class AbstractClassLoaderValue<CLV extends AbstractClassLoaderValue<CLV, V>, V> {

    @Positive
    public abstract Object key();

    @Positive
    public <K> Sub<K> sub(K key);

    @Positive
    public abstract boolean isEqualOrDescendantOf(AbstractClassLoaderValue<?, V> clv);

    @Positive
    public V get(ClassLoader cl);

    @Positive
    public V putIfAbsent(ClassLoader cl, V v);

    @Positive
    public boolean remove(ClassLoader cl, Object v);

    @Positive
    @PolyNull
    @Positive
    public V computeIfAbsent(ClassLoader cl, BiFunction<? super ClassLoader, ? super CLV, ? extends @PolyNull V> mappingFunction) throws IllegalStateException;

    @Positive
    public void removeAll(ClassLoader cl);

    @Positive
    private static final class Memoizer<CLV extends AbstractClassLoaderValue<CLV, V>, V> implements Supplier<V> {

    @Positive
        @Override
    @Positive
        public V get() throws RecursiveInvocationException;

    @Positive
        static class RecursiveInvocationException extends IllegalStateException {
    @Positive
        }
    @Positive
    }

    @Positive
    public final class Sub<K> extends AbstractClassLoaderValue<Sub<K>, V> {

    @Positive
        public AbstractClassLoaderValue<CLV, V> parent();

    @Positive
        @Override
    @Positive
        public K key();

    @Positive
        @Override
    @Positive
        public boolean isEqualOrDescendantOf(AbstractClassLoaderValue<?, V> clv);

    @Positive
        @Override
    @Positive
        public boolean equals(Object o);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }
    @Positive
}
