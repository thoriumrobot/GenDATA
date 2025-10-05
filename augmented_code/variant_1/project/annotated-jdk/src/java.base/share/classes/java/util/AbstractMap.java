/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresKeyFor;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresKeyForIf;
    @Positive
import org.checkerframework.checker.nullness.qual.KeyFor;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.util.Map.Entry;

    @Positive
@CFComment("lock: Subclasses of this interface/class may opt to prohibit null elements")
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public abstract class AbstractMap<K, V> implements Map<K, V> {

    @Positive
    @SideEffectFree
    @Positive
    protected AbstractMap() {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int size(@GuardSatisfied AbstractMap<K, V> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty(@GuardSatisfied AbstractMap<K, V> this);

    @Positive
    @Pure
    @Positive
    public boolean containsValue(@GuardSatisfied AbstractMap<K, V> this, @GuardSatisfied @UnknownSignedness Object value);

    @Positive
    @EnsuresKeyForIf(expression = { "#1" }, result = true, map = { "this" })
    @Positive
    @Pure
    @Positive
    public boolean containsKey(@GuardSatisfied AbstractMap<K, V> this, @GuardSatisfied @UnknownSignedness Object key);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public V get(@GuardSatisfied AbstractMap<K, V> this, @UnknownSignedness @GuardSatisfied Object key);

    @Positive
    @ReleasesNoLocks
    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @Nullable
    @Positive
    public V put(@GuardSatisfied AbstractMap<K, V> this, K key, V value);

    @Positive
    @Nullable
    @Positive
    public V remove(@GuardSatisfied AbstractMap<K, V> this, @GuardSatisfied @UnknownSignedness Object key);

    @Positive
    public void putAll(@GuardSatisfied AbstractMap<K, V> this, Map<? extends K, ? extends V> m);

    @Positive
    public void clear(@GuardSatisfied AbstractMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public Set<@KeyFor({ "this" }) K> keySet(@GuardSatisfied AbstractMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public Collection<V> values(@GuardSatisfied AbstractMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public abstract Set<Entry<@KeyFor({ "this" }) K, V>> entrySet(@GuardSatisfied AbstractMap<K, V> this);

    @Positive
    @Pure
    @Positive
    public boolean equals(@GuardSatisfied AbstractMap<K, V> this, @GuardSatisfied @Nullable Object o);

    @Positive
    @Pure
    @Positive
    public int hashCode(@GuardSatisfied AbstractMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied AbstractMap<K, V> this);

    @Positive
    protected Object clone() throws CloneNotSupportedException;

    @Positive
    public static class SimpleEntry<K, V> implements Entry<K, V>, java.io.Serializable {

    @Positive
        public SimpleEntry(K key, V value) {
    @Positive
        }

    @Positive
        public SimpleEntry(Entry<? extends K, ? extends V> entry) {
    @Positive
        }

    @Positive
        @Pure
    @Positive
        public K getKey(AbstractMap.@GuardSatisfied SimpleEntry<K, V> this);

    @Positive
        @Pure
    @Positive
        public V getValue(AbstractMap.@GuardSatisfied SimpleEntry<K, V> this);

    @Positive
        public V setValue(AbstractMap.@GuardSatisfied SimpleEntry<K, V> this, V value);

    @Positive
        @Pure
    @Positive
        public boolean equals(AbstractMap.@GuardSatisfied SimpleEntry<K, V> this, @GuardSatisfied @Nullable Object o);

    @Positive
        @Pure
    @Positive
        public int hashCode(AbstractMap.@GuardSatisfied SimpleEntry<K, V> this);

    @Positive
        @SideEffectFree
    @Positive
        public String toString(AbstractMap.@GuardSatisfied SimpleEntry<K, V> this);
    @Positive
    }

    @Positive
    public static class SimpleImmutableEntry<K, V> implements Entry<K, V>, java.io.Serializable {

    @Positive
        public SimpleImmutableEntry(K key, V value) {
    @Positive
        }

    @Positive
        public SimpleImmutableEntry(Entry<? extends K, ? extends V> entry) {
    @Positive
        }

    @Positive
        @Pure
    @Positive
        public K getKey(AbstractMap.@GuardSatisfied SimpleImmutableEntry<K, V> this);

    @Positive
        @Pure
    @Positive
        public V getValue(AbstractMap.@GuardSatisfied SimpleImmutableEntry<K, V> this);

    @Positive
        public V setValue(AbstractMap.@GuardSatisfied SimpleImmutableEntry<K, V> this, V value);

    @Positive
        @Pure
    @Positive
        public boolean equals(AbstractMap.@GuardSatisfied SimpleImmutableEntry<K, V> this, @GuardSatisfied @Nullable Object o);

    @Positive
        @Pure
    @Positive
        public int hashCode(AbstractMap.@GuardSatisfied SimpleImmutableEntry<K, V> this);

    @Positive
        @SideEffectFree
    @Positive
        public String toString(AbstractMap.@GuardSatisfied SimpleImmutableEntry<K, V> this);
    @Positive
    }
    @Positive
}
