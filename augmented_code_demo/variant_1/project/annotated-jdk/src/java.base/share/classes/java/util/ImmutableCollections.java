/*
    @Positive
 * Copyright (c) 2016, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamException;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.function.UnaryOperator;
    @Positive
import jdk.internal.access.JavaUtilCollectionAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.misc.CDS;
    @Positive
import jdk.internal.vm.annotation.Stable;

    @Positive
@SuppressWarnings("serial")
    @Positive
class ImmutableCollections {

    @Positive
    static class Access {
    @Positive
    }

    @Positive
    static UnsupportedOperationException uoe();

    @Positive
    @jdk.internal.ValueBased
    @Positive
    static abstract class AbstractImmutableCollection<E> extends AbstractCollection<E> {

    @Positive
        @Override
    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(E e);

    @Positive
        @Override
    @Positive
        public boolean addAll(Collection<? extends E> c);

    @Positive
        @Override
    @Positive
        public void clear();

    @Positive
        @Override
    @Positive
        public boolean remove(@UnknownSignedness Object o);

    @Positive
        @Override
    @Positive
        public boolean removeAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        @Override
    @Positive
        public boolean removeIf(Predicate<? super E> filter);

    @Positive
        @Override
    @Positive
        public boolean retainAll(Collection<? extends @UnknownSignedness Object> c);
    @Positive
    }

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <E> List<E> listCopy(Collection<? extends E> coll);

    @Positive
    @SafeVarargs
    @Positive
    static <E> List<E> listFromArray(E... input);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <E> List<E> listFromTrustedArray(Object... input);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <E> List<E> listFromTrustedArrayNullsAllowed(Object... input);

    @Positive
    @jdk.internal.ValueBased
    @Positive
    static abstract class AbstractImmutableList<E> extends AbstractImmutableCollection<E> implements List<E>, RandomAccess {

    @Positive
        @Override
    @Positive
        public void add(int index, E element);

    @Positive
        @Override
    @Positive
        public boolean addAll(int index, Collection<? extends E> c);

    @Positive
        @Override
    @Positive
        public E remove(int index);

    @Positive
        @Override
    @Positive
        public void replaceAll(UnaryOperator<E> operator);

    @Positive
        @Override
    @Positive
        public E set(int index, E element);

    @Positive
        @Override
    @Positive
        public void sort(Comparator<? super E> c);

    @Positive
        @Override
    @Positive
        public List<E> subList(int fromIndex, int toIndex);

    @Positive
        static void subListRangeCheck(int fromIndex, int toIndex, int size);

    @Positive
        @Override
    @Positive
        public Iterator<E> iterator();

    @Positive
        @Override
    @Positive
        public ListIterator<E> listIterator();

    @Positive
        @Override
    @Positive
        public ListIterator<E> listIterator(final int index);

    @Positive
        @Override
    @Positive
        public boolean equals(Object o);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@UnknownSignedness Object o);

    @Positive
        IndexOutOfBoundsException outOfBounds(int index);
    @Positive
    }

    @Positive
    static final class ListItr<E> implements ListIterator<E> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty ListItr<E> this);

    @Positive
        public void remove();

    @Positive
        public boolean hasPrevious();

    @Positive
        public E previous();

    @Positive
        public int nextIndex();

    @Positive
        public int previousIndex();

    @Positive
        public void set(E e);

    @Positive
        public void add(E e);
    @Positive
    }

    @Positive
    static final class SubList<E> extends AbstractImmutableList<E> implements RandomAccess {

    @Positive
        static <E> SubList<E> fromSubList(SubList<E> parent, int fromIndex, int toIndex);

    @Positive
        static <E> SubList<E> fromList(AbstractImmutableList<E> list, int fromIndex, int toIndex);

    @Positive
        public E get(int index);

    @Positive
        @Pure
    @Positive
        public int size();

    @Positive
        public Iterator<E> iterator();

    @Positive
        public ListIterator<E> listIterator(int index);

    @Positive
        public List<E> subList(int fromIndex, int toIndex);

    @Positive
        @Override
    @Positive
        public int indexOf(Object o);

    @Positive
        @Override
    @Positive
        public int lastIndexOf(Object o);

    @Positive
        @Override
    @Positive
        public Object[] toArray();

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);
    @Positive
    }

    @Positive
    @jdk.internal.ValueBased
    @Positive
    static final class List12<E> extends AbstractImmutableList<E> implements Serializable {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public int size();

    @Positive
        @Override
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public boolean isEmpty();

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public E get(int index);

    @Positive
        @Override
    @Positive
        public int indexOf(Object o);

    @Positive
        @Override
    @Positive
        public int lastIndexOf(Object o);

    @Positive
        @Override
    @Positive
        public Object[] toArray();

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);
    @Positive
    }

    @Positive
    @jdk.internal.ValueBased
    @Positive
    static final class ListN<E> extends AbstractImmutableList<E> implements Serializable {

    @Positive
        @Pure
    @Positive
        @Override
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public boolean isEmpty();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public int size();

    @Positive
        @Override
    @Positive
        public E get(int index);

    @Positive
        @Override
    @Positive
        public Object[] toArray();

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        @Override
    @Positive
        public int indexOf(Object o);

    @Positive
        @Override
    @Positive
        public int lastIndexOf(Object o);
    @Positive
    }

    @Positive
    @jdk.internal.ValueBased
    @Positive
    static abstract class AbstractImmutableSet<E> extends AbstractImmutableCollection<E> implements Set<E> {

    @Positive
        @Override
    @Positive
        public boolean equals(Object o);

    @Positive
        @Override
    @Positive
        public abstract int hashCode();
    @Positive
    }

    @Positive
    @jdk.internal.ValueBased
    @Positive
    static final class Set12<E> extends AbstractImmutableSet<E> implements Serializable {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public int size();

    @Positive
        @Override
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public boolean isEmpty();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@UnknownSignedness Object o);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public Iterator<E> iterator();

    @Positive
        @Override
    @Positive
        public Object[] toArray();

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);
    @Positive
    }

    @Positive
    @jdk.internal.ValueBased
    @Positive
    static final class SetN<E> extends AbstractImmutableSet<E> implements Serializable {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public int size();

    @Positive
        @Override
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public boolean isEmpty();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@UnknownSignedness Object o);

    @Positive
        private final class SetNIterator implements Iterator<E> {

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
            public E next(@NonEmpty SetNIterator this);
    @Positive
        }

    @Positive
        @Override
    @Positive
        public Iterator<E> iterator();

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public Object[] toArray();

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);
    @Positive
    }

    @Positive
    @jdk.internal.ValueBased
    @Positive
    abstract static class AbstractImmutableMap<K, V> extends AbstractMap<K, V> implements Serializable {

    @Positive
        @Override
    @Positive
        public void clear();

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V compute(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> rf);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V computeIfAbsent(K key, Function<? super K, ? extends @PolyNull V> mf);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V computeIfPresent(K key, BiFunction<? super K, ? super V, ? extends @PolyNull V> rf);

    @Positive
        @Override
    @Positive
        @PolyNull
    @Positive
        public V merge(K key, @NonNull V value, BiFunction<? super V, ? super V, ? extends @PolyNull V> rf);

    @Positive
        @Override
    @Positive
        public V put(K key, V value);

    @Positive
        @Override
    @Positive
        public void putAll(Map<? extends K, ? extends V> m);

    @Positive
        @Override
    @Positive
        public V putIfAbsent(K key, V value);

    @Positive
        @Override
    @Positive
        public V remove(Object key);

    @Positive
        @Override
    @Positive
        public boolean remove(@UnknownSignedness Object key, @UnknownSignedness Object value);

    @Positive
        @Override
    @Positive
        public V replace(K key, V value);

    @Positive
        @Override
    @Positive
        public boolean replace(K key, V oldValue, V newValue);

    @Positive
        @Override
    @Positive
        public void replaceAll(BiFunction<? super K, ? super V, ? extends V> f);

    @Positive
        @Override
    @Positive
        public V getOrDefault(Object key, V defaultValue);
    @Positive
    }

    @Positive
    @jdk.internal.ValueBased
    @Positive
    static final class Map1<K, V> extends AbstractImmutableMap<K, V> {

    @Positive
        @Override
    @Positive
        @SideEffectFree
    @Positive
        public Set<Map.Entry<K, V>> entrySet();

    @Positive
        @Override
    @Positive
        public V get(Object o);

    @Positive
        @Pure
    @Positive
        @Override
    @Positive
        public boolean containsKey(@UnknownSignedness Object o);

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public boolean containsValue(@UnknownSignedness Object o);

    @Positive
        @Override
    @Positive
        public int size();

    @Positive
        @Override
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public boolean isEmpty();

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    @jdk.internal.ValueBased
    @Positive
    static final class MapN<K, V> extends AbstractImmutableMap<K, V> {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public boolean containsKey(@UnknownSignedness Object o);

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public boolean containsValue(@UnknownSignedness Object o);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public V get(Object o);

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public int size();

    @Positive
        @Override
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public boolean isEmpty();

    @Positive
        class MapNIterator implements Iterator<Map.Entry<K, V>> {

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
            public Map.Entry<K, V> next(@NonEmpty MapNIterator this);
    @Positive
        }

    @Positive
        @Override
    @Positive
        @SideEffectFree
    @Positive
        public Set<Map.Entry<K, V>> entrySet();
    @Positive
    }
    @Positive
}

    @Positive
final class CollSer implements Serializable {
    @Positive
}

// CFWR semantic augmentation - variant 1
