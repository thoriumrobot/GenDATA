/*
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.checker.nullness.qual.EnsuresKeyFor;
import org.checkerframework.checker.nullness.qual.EnsuresKeyForIf;
import org.checkerframework.checker.nullness.qual.KeyFor;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.util.Map.Entry;

@CFComment("lock: Subclasses of this interface/class may opt to prohibit null elements")
@AnnotatedFor({ "lock", "nullness", "index" })
public abstract class AbstractMap<K, V> implements Map<K, V> {

    @SideEffectFree
    protected AbstractMap() {
    }

    @Pure
    @NonNegative
    public int size(@GuardSatisfied AbstractMap<K, V> this);

    @Pure
    @EnsuresNonEmptyIf(result = false, expression = "this")
    public boolean isEmpty(@GuardSatisfied AbstractMap<K, V> this);

    @Pure
    public boolean containsValue(@GuardSatisfied AbstractMap<K, V> this, @GuardSatisfied @UnknownSignedness Object value);

    @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Pure
    public boolean containsKey(@GuardSatisfied AbstractMap<K, V> this, @GuardSatisfied @UnknownSignedness Object key);

    @Pure
    @Nullable
    public V get(@GuardSatisfied AbstractMap<K, V> this, @UnknownSignedness @GuardSatisfied Object key);

    @ReleasesNoLocks
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Nullable
    public V put(@GuardSatisfied AbstractMap<K, V> this, K key, V value);

    @Nullable
    public V remove(@GuardSatisfied AbstractMap<K, V> this, @GuardSatisfied @UnknownSignedness Object key);

    public void putAll(@GuardSatisfied AbstractMap<K, V> this, Map<? extends K, ? extends V> m);

    public void clear(@GuardSatisfied AbstractMap<K, V> this);

    @SideEffectFree
    public Set<@KeyFor({ "this" }) K> keySet(@GuardSatisfied AbstractMap<K, V> this);

    @SideEffectFree
    public Collection<V> values(@GuardSatisfied AbstractMap<K, V> this);

    @SideEffectFree
    public abstract Set<Entry<@KeyFor({ "this" }) K, V>> entrySet(@GuardSatisfied AbstractMap<K, V> this);

    @Pure
    public boolean equals(@GuardSatisfied AbstractMap<K, V> this, @GuardSatisfied @Nullable Object o);

    @Pure
    public int hashCode(@GuardSatisfied AbstractMap<K, V> this);

    @SideEffectFree
    public String toString(@GuardSatisfied AbstractMap<K, V> this);

    protected Object clone() throws CloneNotSupportedException;

    public static class SimpleEntry<K, V> implements Entry<K, V>, java.io.Serializable {

        public SimpleEntry(K key, V value) {
        }

        public SimpleEntry(Entry<? extends K, ? extends V> entry) {
        }

        @Pure
        public K getKey(AbstractMap.@GuardSatisfied SimpleEntry<K, V> this);

        @Pure
        public V getValue(AbstractMap.@GuardSatisfied SimpleEntry<K, V> this);

        public V setValue(AbstractMap.@GuardSatisfied SimpleEntry<K, V> this, V value);

        @Pure
        public boolean equals(AbstractMap.@GuardSatisfied SimpleEntry<K, V> this, @GuardSatisfied @Nullable Object o);

        @Pure
        public int hashCode(AbstractMap.@GuardSatisfied SimpleEntry<K, V> this);

        @SideEffectFree
        public String toString(AbstractMap.@GuardSatisfied SimpleEntry<K, V> this);
    }

    public static class SimpleImmutableEntry<K, V> implements Entry<K, V>, java.io.Serializable {

        public SimpleImmutableEntry(K key, V value) {
        }

        public SimpleImmutableEntry(Entry<? extends K, ? extends V> entry) {
        }

        @Pure
        public K getKey(AbstractMap.@GuardSatisfied SimpleImmutableEntry<K, V> this);

        @Pure
        public V getValue(AbstractMap.@GuardSatisfied SimpleImmutableEntry<K, V> this);

        public V setValue(AbstractMap.@GuardSatisfied SimpleImmutableEntry<K, V> this, V value);

        @Pure
        public boolean equals(AbstractMap.@GuardSatisfied SimpleImmutableEntry<K, V> this, @GuardSatisfied @Nullable Object o);

        @Pure
        public int hashCode(AbstractMap.@GuardSatisfied SimpleImmutableEntry<K, V> this);

        @SideEffectFree
        public String toString(AbstractMap.@GuardSatisfied SimpleImmutableEntry<K, V> this);
    }
}
