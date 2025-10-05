/*
 * Copyright (c) 2016, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.Collection;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;

final class WeakPairMap<K1, K2, V> {

    public boolean containsKeyPair(K1 k1, K2 k2);

    public V get(K1 k1, K2 k2);

    public V put(K1 k1, K2 k2, V v);

    public V putIfAbsent(K1 k1, K2 k2, V v);

    @PolyNull
    public V computeIfAbsent(K1 k1, K2 k2, BiFunction<? super K1, ? super K2, ? extends @PolyNull V> mappingFunction);

    public Collection<V> values();

    private interface Pair<K1, K2> {

        static <K1, K2> Pair<K1, K2> weak(K1 k1, K2 k2, ReferenceQueue<Object> queue);

        static <K1, K2> Pair<K1, K2> lookup(K1 k1, K2 k2);

        K1 first();

        K2 second();

        static int hashCode(Object first, Object second);

        static boolean equals(Object first, Object second, Pair<?, ?> p);

        final class Weak<K1, K2> extends WeakRefPeer<K1> implements Pair<K1, K2> {

            private final int hash;

            private final WeakRefPeer<K2> peer;

            Weak(K1 k1, K2 k2, ReferenceQueue<Object> queue) {
            }

            @Override
            Weak<?, ?> weakPair();

            @Override
            public K1 first();

            @Override
            public K2 second();

            @Override
            public int hashCode();

            @Override
            public boolean equals(Object obj);
        }

        final class Lookup<K1, K2> implements Pair<K1, K2> {

            private final K1 k1;

            private final K2 k2;

            Lookup(K1 k1, K2 k2) {
            }

            @Override
            public K1 first();

            @Override
            public K2 second();

            @Override
            public int hashCode();

            @Override
            public boolean equals(Object obj);
        }
    }

    private static abstract class WeakRefPeer<K> extends WeakReference<K> {

        abstract Pair.Weak<?, ?> weakPair();
    }
}
