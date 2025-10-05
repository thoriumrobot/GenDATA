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
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
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
import java.util.function.Consumer;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.function.UnaryOperator;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.util.ArraysSupport;

    @Positive
@CFComment("lock/nullness: Permit null elements")
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class ArrayList<E> extends AbstractList<E> implements List<E>, RandomAccess, Cloneable, java.io.Serializable {

    @Positive
    @SideEffectFree
    @Positive
    public ArrayList(@NonNegative int initialCapacity) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public ArrayList() {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    @PolyNonEmpty
    @Positive
    public ArrayList(@PolyNonEmpty Collection<? extends E> c) {
    @Positive
    }

    @Positive
    public void trimToSize(@GuardSatisfied ArrayList<E> this);

    @Positive
    public void ensureCapacity(@GuardSatisfied ArrayList<E> this, int minCapacity);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int size(@GuardSatisfied ArrayList<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty(@GuardSatisfied ArrayList<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied ArrayList<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int indexOf(@GuardSatisfied ArrayList<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    int indexOfRange(@GuardSatisfied @Nullable @UnknownSignedness Object o, int start, int end);

    @Positive
    @Pure
    @Positive
    @GTENegativeOne
    @Positive
    public int lastIndexOf(@GuardSatisfied ArrayList<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    int lastIndexOfRange(@GuardSatisfied @Nullable @UnknownSignedness Object o, int start, int end);

    @Positive
    @SideEffectFree
    @Positive
    public Object clone(@GuardSatisfied ArrayList<E> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public Object[] toArray(ArrayList<@PolyNull @PolySigned E> this);

    @Positive
    @SideEffectFree
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public <T> T[] toArray(@PolyNull T[] a);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    E elementData(@NonNegative int index);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <E> E elementAt(Object[] es, int index);

    @Positive
    @Pure
    @Positive
    public E get(@GuardSatisfied ArrayList<E> this, @NonNegative int index);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public E set(@GuardSatisfied ArrayList<E> this, @NonNegative int index, E element);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public boolean add(@GuardSatisfied ArrayList<E> this, E e);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public void add(@GuardSatisfied ArrayList<E> this, @NonNegative int index, E element);

    @Positive
    public E remove(@GuardSatisfied @Shrinkable ArrayList<E> this, @NonNegative int index);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    boolean equalsRange(List<?> other, int from, int to);

    @Positive
    public int hashCode();

    @Positive
    int hashCodeRange(int from, int to);

    @Positive
    public boolean remove(@GuardSatisfied @Shrinkable ArrayList<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public void clear(@GuardSatisfied @Shrinkable ArrayList<E> this);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public boolean addAll(@GuardSatisfied ArrayList<E> this, Collection<? extends E> c);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public boolean addAll(@GuardSatisfied ArrayList<E> this, @NonNegative int index, Collection<? extends E> c);

    @Positive
    protected void removeRange(@GuardSatisfied @Shrinkable ArrayList<E> this, int fromIndex, int toIndex);

    @Positive
    public boolean removeAll(@Shrinkable ArrayList<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    public boolean retainAll(@GuardSatisfied @Shrinkable ArrayList<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    boolean batchRemove(Collection<?> c, boolean complement, final int from, final int end);

    @Positive
    @PolyGrowShrink
    @Positive
    public ListIterator<E> listIterator(@PolyGrowShrink ArrayList<E> this, @NonNegative int index);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public ListIterator<E> listIterator(@PolyGrowShrink @PolyNonEmpty ArrayList<E> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty ArrayList<E> this);

    @Positive
    private class Itr implements Iterator<E> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty Itr this);

    @Positive
        public void remove();

    @Positive
        @SuppressWarnings({ "unchecked" })
    @Positive
        @Override
    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        final void checkForComodification();
    @Positive
    }

    @Positive
    private class ListItr extends Itr implements ListIterator<E> {

    @Positive
        public boolean hasPrevious();

    @Positive
        public int nextIndex();

    @Positive
        public int previousIndex();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public E previous();

    @Positive
        public void set(E e);

    @Positive
        public void add(E e);
    @Positive
    }

    @Positive
    @PolyGrowShrink
    @Positive
    public List<E> subList(@GuardSatisfied @PolyGrowShrink ArrayList<E> this, @NonNegative int fromIndex, @NonNegative int toIndex);

    @Positive
    private static class SubList<E> extends AbstractList<E> implements RandomAccess {

    @Positive
        public SubList(ArrayList<E> root, int fromIndex, int toIndex) {
    @Positive
        }

    @Positive
        public E set(@NonNegative int index, E element);

    @Positive
        public E get(@NonNegative int index);

    @Positive
        @Pure
    @Positive
        @NonNegative
    @Positive
        public int size();

    @Positive
        public void add(@NonNegative int index, E element);

    @Positive
        public E remove(@NonNegative int index);

    @Positive
        protected void removeRange(int fromIndex, int toIndex);

    @Positive
        public boolean addAll(Collection<? extends E> c);

    @Positive
        public boolean addAll(@NonNegative int index, Collection<? extends E> c);

    @Positive
        public void replaceAll(UnaryOperator<E> operator);

    @Positive
        public boolean removeAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        public boolean retainAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        public boolean removeIf(Predicate<? super E> filter);

    @Positive
        @PolyNull
    @Positive
        @PolySigned
    @Positive
        public Object[] toArray(SubList<@PolyNull @PolySigned E> this);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(T[] a);

    @Positive
        public boolean equals(@Nullable Object o);

    @Positive
        public int hashCode();

    @Positive
        public int indexOf(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
        public int lastIndexOf(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@Nullable @UnknownSignedness Object o);

    @Positive
        @SideEffectFree
    @Positive
        public Iterator<E> iterator();

    @Positive
        public ListIterator<E> listIterator(@NonNegative int index);

    @Positive
        public List<E> subList(int fromIndex, int toIndex);

    @Positive
        @SideEffectFree
    @Positive
        public Spliterator<E> spliterator();
    @Positive
    }

    @Positive
    @Override
    @Positive
    public void forEach(Consumer<? super E> action);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public Spliterator<E> spliterator();

    @Positive
    final class ArrayListSpliterator implements Spliterator<E> {

    @Positive
        public ArrayListSpliterator trySplit();

    @Positive
        public boolean tryAdvance(Consumer<? super E> action);

    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        public long estimateSize();

    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    @Override
    @Positive
    public boolean removeIf(@Shrinkable ArrayList<E> this, Predicate<? super E> filter);

    @Positive
    boolean removeIf(@Shrinkable ArrayList<E> this, Predicate<? super E> filter, int i, final int end);

    @Positive
    @SuppressWarnings({ "unchecked" })
    @Positive
    @Override
    @Positive
    public void replaceAll(UnaryOperator<E> operator);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public void sort(Comparator<? super E> c);

    @Positive
    void checkInvariants();
    @Positive
}
