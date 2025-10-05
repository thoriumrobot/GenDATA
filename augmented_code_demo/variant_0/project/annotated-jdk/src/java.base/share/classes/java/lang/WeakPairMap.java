/*
    @Positive
 * Copyright (c) 2016, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import java.lang.ref.Reference;
    @Positive
import java.lang.ref.ReferenceQueue;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Objects;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.function.BiFunction;

    @Positive
final class WeakPairMap<K1, K2, V> {

    @Positive
    public boolean containsKeyPair(K1 k1, K2 k2);

    @Positive
    public V get(K1 k1, K2 k2);

    @Positive
    public V put(K1 k1, K2 k2, V v);

    @Positive
    public V putIfAbsent(K1 k1, K2 k2, V v);

    @Positive
    @PolyNull
    @Positive
    public V computeIfAbsent(K1 k1, K2 k2, BiFunction<? super K1, ? super K2, ? extends @PolyNull V> mappingFunction);

    @Positive
    public Collection<V> values();

    @Positive
    private interface Pair<K1, K2> {

    @Positive
        static <K1, K2> Pair<K1, K2> weak(K1 k1, K2 k2, ReferenceQueue<Object> queue);

    @Positive
        static <K1, K2> Pair<K1, K2> lookup(K1 k1, K2 k2);

    @Positive
        K1 first();

    @Positive
        K2 second();

    @Positive
        static int hashCode(Object first, Object second);

    @Positive
        static boolean equals(Object first, Object second, Pair<?, ?> p);

    @Positive
        final class Weak<K1, K2> extends WeakRefPeer<K1> implements Pair<K1, K2> {

    @Positive
            private final int hash;

    @Positive
            private final WeakRefPeer<K2> peer;

    @Positive
            Weak(K1 k1, K2 k2, ReferenceQueue<Object> queue) {
    @Positive
            }

    @Positive
            @Override
    @Positive
            Weak<?, ?> weakPair();

    @Positive
            @Override
    @Positive
            public K1 first();

    @Positive
            @Override
    @Positive
            public K2 second();

    @Positive
            @Override
    @Positive
            public int hashCode();

    @Positive
            @Override
    @Positive
            public boolean equals(Object obj);
    @Positive
        }

    @Positive
        final class Lookup<K1, K2> implements Pair<K1, K2> {

    @Positive
            private final K1 k1;

    @Positive
            private final K2 k2;

    @Positive
            Lookup(K1 k1, K2 k2) {
    @Positive
            }

    @Positive
            @Override
    @Positive
            public K1 first();

    @Positive
            @Override
    @Positive
            public K2 second();

    @Positive
            @Override
    @Positive
            public int hashCode();

    @Positive
            @Override
    @Positive
            public boolean equals(Object obj);
    @Positive
        }
    @Positive
    }

    @Positive
    private static abstract class WeakRefPeer<K> extends WeakReference<K> {

    @Positive
        abstract Pair.Weak<?, ?> weakPair();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
