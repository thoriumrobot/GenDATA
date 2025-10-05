/*
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
package java.util.concurrent;

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
import java.lang.invoke.VarHandle;
    @Positive
import java.lang.reflect.Field;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.ConcurrentModificationException;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.ListIterator;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.RandomAccess;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.function.UnaryOperator;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
public class CopyOnWriteArrayList<E> implements List<E>, RandomAccess, Cloneable, java.io.Serializable {

    @Positive
    final Object[] getArray();

    @Positive
    final void setArray(Object[] a);

    @Positive
    public CopyOnWriteArrayList() {
    @Positive
    }

    @Positive
    public CopyOnWriteArrayList(Collection<? extends E> c) {
    @Positive
    }

    @Positive
    public CopyOnWriteArrayList(E[] toCopyIn) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public int size();

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty();

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public int indexOf(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public int indexOf(E e, int index);

    @Positive
    public int lastIndexOf(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public int lastIndexOf(E e, int index);

    @Positive
    public Object clone();

    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public Object[] toArray(CopyOnWriteArrayList<@PolyNull @PolySigned E> this);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public <T> T[] toArray(@PolyNull T[] a);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <E> E elementAt(Object[] a, int index);

    @Positive
    static String outOfBounds(int index, int size);

    @Positive
    public E get(int index);

    @Positive
    public E set(int index, E element);

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public boolean add(E e);

    @Positive
    public void add(int index, E element);

    @Positive
    public E remove(@GuardSatisfied @Shrinkable CopyOnWriteArrayList<E> this, int index);

    @Positive
    public boolean remove(@Shrinkable CopyOnWriteArrayList<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    void removeRange(@GuardSatisfied @Shrinkable CopyOnWriteArrayList<E> this, int fromIndex, int toIndex);

    @Positive
    public boolean addIfAbsent(E e);

    @Positive
    @Pure
    @Positive
    public boolean containsAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
    public boolean removeAll(@Shrinkable CopyOnWriteArrayList<E> this, Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    public boolean retainAll(@GuardSatisfied @Shrinkable CopyOnWriteArrayList<E> this, Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    public int addAllAbsent(Collection<? extends E> c);

    @Positive
    public void clear(@GuardSatisfied @Shrinkable CopyOnWriteArrayList<E> this);

    @Positive
    public boolean addAll(Collection<? extends E> c);

    @Positive
    public boolean addAll(int index, Collection<? extends E> c);

    @Positive
    public void forEach(Consumer<? super E> action);

    @Positive
    public boolean removeIf(@Shrinkable CopyOnWriteArrayList<E> this, Predicate<? super E> filter);

    @Positive
    boolean bulkRemove(Predicate<? super E> filter, int i, int end);

    @Positive
    public void replaceAll(UnaryOperator<E> operator);

    @Positive
    void replaceAllRange(UnaryOperator<E> operator, int i, int end);

    @Positive
    public void sort(Comparator<? super E> c);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    void sortRange(Comparator<? super E> c, int i, int end);

    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public int hashCode();

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty CopyOnWriteArrayList<E> this);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public ListIterator<E> listIterator(@PolyGrowShrink @PolyNonEmpty CopyOnWriteArrayList<E> this);

    @Positive
    @PolyGrowShrink
    @Positive
    public ListIterator<E> listIterator(@PolyGrowShrink CopyOnWriteArrayList<E> this, int index);

    @Positive
    public Spliterator<E> spliterator();

    @Positive
    static final class COWIterator<E> implements ListIterator<E> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @Pure
    @Positive
        public boolean hasPrevious();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty COWIterator<E> this);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E previous();

    @Positive
        @Pure
    @Positive
        public int nextIndex();

    @Positive
        @Pure
    @Positive
        public int previousIndex();

    @Positive
        public void remove();

    @Positive
        public void set(E e);

    @Positive
        public void add(E e);

    @Positive
        @Override
    @Positive
        public void forEachRemaining(Consumer<? super E> action);
    @Positive
    }

    @Positive
    @PolyGrowShrink
    @Positive
    public List<E> subList(@PolyGrowShrink CopyOnWriteArrayList<E> this, int fromIndex, int toIndex);

    @Positive
    private class COWSubList implements List<E>, RandomAccess {

    @Positive
        public Object[] toArray();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public int indexOf(Object o);

    @Positive
        public int lastIndexOf(Object o);

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@Nullable @UnknownSignedness Object o);

    @Positive
        @Pure
    @Positive
        public boolean containsAll(Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public boolean isEmpty();

    @Positive
        public String toString();

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object o);

    @Positive
        public E set(int index, E element);

    @Positive
        public E get(int index);

    @Positive
        @Pure
    @Positive
        public int size();

    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(E element);

    @Positive
        public void add(int index, E element);

    @Positive
        public boolean addAll(Collection<? extends E> c);

    @Positive
        public boolean addAll(int index, Collection<? extends E> c);

    @Positive
        public void clear();

    @Positive
        public E remove(int index);

    @Positive
        public boolean remove(@Nullable @UnknownSignedness Object o);

    @Positive
        public Iterator<E> iterator();

    @Positive
        public ListIterator<E> listIterator();

    @Positive
        public ListIterator<E> listIterator(int index);

    @Positive
        public List<E> subList(int fromIndex, int toIndex);

    @Positive
        public void forEach(Consumer<? super E> action);

    @Positive
        public void replaceAll(UnaryOperator<E> operator);

    @Positive
        public void sort(Comparator<? super E> c);

    @Positive
        public boolean removeAll(Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
        public boolean retainAll(Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
        public boolean removeIf(Predicate<? super E> filter);

    @Positive
        public Spliterator<E> spliterator();
    @Positive
    }

    @Positive
    private static class COWSubListIterator<E> implements ListIterator<E> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty COWSubListIterator<E> this);

    @Positive
        public boolean hasPrevious();

    @Positive
        public E previous();

    @Positive
        public int nextIndex();

    @Positive
        public int previousIndex();

    @Positive
        public void remove();

    @Positive
        public void set(E e);

    @Positive
        public void add(E e);

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public void forEachRemaining(Consumer<? super E> action);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
