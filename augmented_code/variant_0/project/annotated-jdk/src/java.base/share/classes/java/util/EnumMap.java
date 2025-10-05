/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.initialization.qual.UnknownInitialization;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresKeyFor;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresKeyForIf;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.nullness.qual.RequiresNonNull;
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
import jdk.internal.access.SharedSecrets;

    @Positive
@AnnotatedFor({ "nullness", "index" })
    @Positive
public class EnumMap<K extends Enum<K>, V> extends AbstractMap<K, V> implements java.io.Serializable, Cloneable {

    @Positive
    public EnumMap(Class<K> keyType) {
    @Positive
    }

    @Positive
    public EnumMap(EnumMap<K, ? extends V> m) {
    @Positive
    }

    @Positive
    public EnumMap(Map<K, ? extends V> m) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int size();

    @Positive
    @Pure
    @Positive
    public boolean containsValue(@GuardSatisfied @Nullable @UnknownSignedness Object value);

    @Positive
    @EnsuresKeyForIf(if ({ "#1" }, result = true, map = { "this" })
    @Positive
    @Pure
    @Positive
    public boolean containsKey(@GuardSatisfied @UnknownSignedness Object key);

    @Positive
    @Nullable
    @Positive
    public V get(@UnknownSignedness @Nullable Object key);

    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @Nullable
    @Positive
    public V put(K key, V value);

    @Positive
    @Nullable
    @Positive
    public V remove(@GuardSatisfied @UnknownSignedness Object key);

    @Positive
    @CFComment({ "nullness: Variables keyUniverse", "and vals are private class members for EnumMap and are absent in AbstractMap." })
    @Positive
    @SuppressWarnings({ "nullness:contracts.precondition.override.invalid" })
    @Positive
    @RequiresNonNull({ "keyUniverse", "vals" })
    @Positive
    public void putAll(@UnknownInitialization EnumMap<K, V> this, Map<) {
            expression = extends K, ? extends V> m);

    @Positive
    public void clear();

    @Positive
    public Set<K> keySet();

    @Positive
    private class KeySet extends AbstractSet<K> {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<K> iterator();

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public int size();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@Nullable @UnknownSignedness Object o);

    @Positive
        public boolean remove(@Nullable @UnknownSignedness Object o);

    @Positive
        public void clear();
    @Positive
    }

    @Positive
    public Collection<V> values();

    @Positive
    private class Values extends AbstractCollection<V> {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<V> iterator();

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public int size();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@Nullable @UnknownSignedness Object o);

    @Positive
        public boolean remove(@Nullable @UnknownSignedness Object o);

    @Positive
        public void clear();
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Set<Map.Entry<K, V>> entrySet();

    @Positive
    private class EntrySet extends AbstractSet<Map.Entry<K, V>> {

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<Map.Entry<K, V>> iterator();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@Nullable @UnknownSignedness Object o);

    @Positive
        public boolean remove(@Nullable @UnknownSignedness Object o);

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public int size();

    @Positive
        public void clear();

    @Positive
        @SideEffectFree
    @Positive
        public Object[] toArray();

    @Positive
        @CFComment({ "nullness;
        } else {
            expression = 'a' is known to be of array class type", "Annotation for toArray are technically incorrect. Refer to note on toArray in Collection.java" })
    @Positive
        @SideEffectFree
    @Positive
        @SuppressWarnings({ "unchecked", "nullness:argument", "nullness:override.param" })
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);
        }
    @Positive
    }

    @Positive
    private abstract class EnumMapIterator<T> implements Iterator<T> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        public void remove();
    @Positive
    }

    @Positive
    private class KeyIterator extends EnumMapIterator<K> {

    @Positive
        public K next(@NonEmpty KeyIterator this);
    @Positive
    }

    @Positive
    private class ValueIterator extends EnumMapIterator<V> {

    @Positive
        @CFComment({ "nullness: Value returned by unmaskNull", "will be of type V (not @Nullable V) for mapped value" })
    @Positive
        @SuppressWarnings({ "nullness:return" })
    @Positive
        public V next(@NonEmpty ValueIterator this);
    @Positive
    }

    @Positive
    private class EntryIterator extends EnumMapIterator<Map.Entry<K, V>> {

    @Positive
        public Map.Entry<K, V> next(@NonEmpty EntryIterator this);

    @Positive
        public void remove();

    @Positive
        private class Entry implements Map.Entry<K, V> {

    @Positive
            public K getKey();

    @Positive
            @CFComment({ "nullness: Value returned by unmaskNull", "will be of type V (not @Nullable V) for mapped value" })
    @Positive
            @SuppressWarnings("nullness:return")
    @Positive
            public V getValue();

    @Positive
            @CFComment({ "nullness: Value returned by unmaskNull", "will be of type V (not @Nullable V) for mapped value" })
    @Positive
            @SuppressWarnings("nullness:return")
    @Positive
            public V setValue(V value);

    @Positive
            public boolean equals(Object o);

    @Positive
            public int hashCode();

    @Positive
            public String toString();
    @Positive
        }
    @Positive
    }

    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public int hashCode();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public EnumMap<K, V> clone();
    @Positive
}
