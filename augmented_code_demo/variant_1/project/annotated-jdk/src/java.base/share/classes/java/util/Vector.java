/*
    @Positive
 * Copyright (c) 1994, 2019, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.PolyGrowShrink;
    @Positive
import org.checkerframework.checker.index.qual.Shrinkable;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.PolyNonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
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
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.StreamCorruptedException;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.function.UnaryOperator;
    @Positive
import jdk.internal.util.ArraysSupport;

    @Positive
@CFComment({ "lock/nullness: permits nullable object" })
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class Vector<E> extends AbstractList<E> implements List<E>, RandomAccess, Cloneable, java.io.Serializable {

    @Positive
    @SuppressWarnings("serial")
    @Positive
    protected Object[] elementData;

    @Positive
    protected int elementCount;

    @Positive
    protected int capacityIncrement;

    @Positive
    public Vector(@NonNegative int initialCapacity, int capacityIncrement) {
    @Positive
    }

    @Positive
    public Vector(@NonNegative int initialCapacity) {
    @Positive
    }

    @Positive
    public Vector() {
    @Positive
    }

    @Positive
    @PolyNonEmpty
    @Positive
    public Vector(@PolyNonEmpty Collection<? extends E> c) {
    @Positive
    }

    @Positive
    public synchronized void copyInto(@Nullable Object[] anArray);

    @Positive
    public synchronized void trimToSize(@GuardSatisfied Vector<E> this);

    @Positive
    public synchronized void ensureCapacity(int minCapacity);

    @Positive
    public synchronized void setSize(@GuardSatisfied @Shrinkable Vector<E> this, @NonNegative int newSize);

    @Positive
    @NonNegative
    @Positive
    public synchronized int capacity();

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public synchronized int size(@GuardSatisfied Vector<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public synchronized boolean isEmpty(@GuardSatisfied Vector<E> this);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Enumeration<E> elements(@PolyGrowShrink @PolyNonEmpty Vector<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied Vector<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int indexOf(@GuardSatisfied Vector<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public synchronized int indexOf(@GuardSatisfied Vector<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o, @NonNegative int index);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public synchronized int lastIndexOf(@GuardSatisfied Vector<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public synchronized int lastIndexOf(@GuardSatisfied Vector<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o, @NonNegative int index);

    @Positive
    public synchronized E elementAt(@NonNegative int index);

    @Positive
    public synchronized E firstElement(@NonEmpty Vector<E> this);

    @Positive
    public synchronized E lastElement(@NonEmpty Vector<E> this);

    @Positive
    public synchronized void setElementAt(@GuardSatisfied Vector<E> this, E obj, @NonNegative int index);

    @Positive
    public synchronized void removeElementAt(@GuardSatisfied @Shrinkable Vector<E> this, @NonNegative int index);

    @Positive
    public synchronized void insertElementAt(@GuardSatisfied Vector<E> this, E obj, @NonNegative int index);

    @Positive
    public synchronized void addElement(@GuardSatisfied Vector<E> this, E obj);

    @Positive
    public synchronized boolean removeElement(@GuardSatisfied @Shrinkable Vector<E> this, Object obj);

    @Positive
    public synchronized void removeAllElements(@GuardSatisfied @Shrinkable Vector<E> this);

    @Positive
    @SideEffectFree
    @Positive
    public synchronized Object clone(@GuardSatisfied Vector<E> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public synchronized Object[] toArray(Vector<@PolyNull @PolySigned E> this);

    @Positive
    @SideEffectFree
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public synchronized <T> T[] toArray(@PolyNull T[] a);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    E elementData(int index);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <E> E elementAt(Object[] es, int index);

    @Positive
    @Pure
    @Positive
    public synchronized E get(@GuardSatisfied Vector<E> this, @NonNegative int index);

    @Positive
    public synchronized E set(@GuardSatisfied Vector<E> this, @NonNegative int index, E element);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public synchronized boolean add(@GuardSatisfied Vector<E> this, E e);

    @Positive
    public boolean remove(@GuardSatisfied @Shrinkable Vector<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public void add(@GuardSatisfied Vector<E> this, @NonNegative int index, E element);

    @Positive
    public synchronized E remove(@GuardSatisfied @Shrinkable Vector<E> this, @NonNegative int index);

    @Positive
    public void clear(@GuardSatisfied @Shrinkable Vector<E> this);

    @Positive
    @Pure
    @Positive
    public synchronized boolean containsAll(@GuardSatisfied Vector<E> this, @GuardSatisfied Collection<? extends @UnknownSignedness Object> c);

    @Positive
    public boolean addAll(@GuardSatisfied Vector<E> this, Collection<? extends E> c);

    @Positive
    public boolean removeAll(@GuardSatisfied @Shrinkable Vector<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    public boolean retainAll(@GuardSatisfied @Shrinkable Vector<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    @SuppressWarnings({ "unchecked" })
    @Positive
    @Override
    @Positive
    public boolean removeIf(@Shrinkable Vector<E> this, Predicate<? super E> filter);

    @Positive
    public synchronized boolean addAll(@GuardSatisfied Vector<E> this, @NonNegative int index, Collection<? extends E> c);

    @Positive
    @Pure
    @Positive
    public synchronized boolean equals(@GuardSatisfied Vector<E> this, @GuardSatisfied @Nullable Object o);

    @Positive
    @Pure
    @Positive
    public synchronized int hashCode(@GuardSatisfied Vector<E> this);

    @Positive
    @SideEffectFree
    @Positive
    public synchronized String toString(@GuardSatisfied Vector<E> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    public synchronized List<E> subList(@GuardSatisfied @PolyGrowShrink Vector<E> this, int fromIndex, int toIndex);

    @Positive
    protected synchronized void removeRange(@GuardSatisfied @Shrinkable Vector<E> this, int fromIndex, int toIndex);

    @Positive
    @PolyGrowShrink
    @Positive
    public synchronized ListIterator<E> listIterator(@PolyGrowShrink Vector<E> this, @NonNegative int index);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public synchronized ListIterator<E> listIterator(@PolyGrowShrink Vector<E> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public synchronized Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty Vector<E> this);

    @Positive
    private class Itr implements Iterator<E> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty Itr this);

    @Positive
        public void remove();

    @Positive
        @Override
    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        final void checkForComodification();
    @Positive
    }

    @Positive
    final class ListItr extends Itr implements ListIterator<E> {

    @Positive
        public boolean hasPrevious();

    @Positive
        public int nextIndex();

    @Positive
        public int previousIndex();

    @Positive
        public E previous();

    @Positive
        public void set(E e);

    @Positive
        public void add(E e);
    @Positive
    }

    @Positive
    @Override
    @Positive
    public synchronized void forEach(Consumer<? super E> action);

    @Positive
    @SuppressWarnings({ "unchecked" })
    @Positive
    @Override
    @Positive
    public synchronized void replaceAll(UnaryOperator<E> operator);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Override
    @Positive
    public synchronized void sort(Comparator<? super E> c);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public Spliterator<E> spliterator();

    @Positive
    final class VectorSpliterator implements Spliterator<E> {

    @Positive
        public Spliterator<E> trySplit();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public boolean tryAdvance(Consumer<? super E> action);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        public long estimateSize();

    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    void checkInvariants();
    @Positive
}

// CFWR semantic augmentation - variant 1
