/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.util;

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
import java.io.Serializable;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Predicate;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public class ArrayDeque<E extends @NonNull Object> extends AbstractCollection<E> implements Deque<E>, Cloneable, Serializable {

    @Positive
    public ArrayDeque() {
    @Positive
    }

    @Positive
    public ArrayDeque(@NonNegative int numElements) {
    @Positive
    }

    @Positive
    @PolyNonEmpty
    @Positive
    public ArrayDeque(@PolyNonEmpty Collection<? extends E> c) {
    @Positive
    }

    @Positive
    static final int inc(int i, int modulus);

    @Positive
    static final int dec(int i, int modulus);

    @Positive
    static final int inc(int i, int distance, int modulus);

    @Positive
    static final int sub(int i, int j, int modulus);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Pure
    @Positive
    static final <E> E elementAt(@PolyNull @PolySigned Object[] es, int i);

    @Positive
    static final <E> E nonNullElementAt(@PolyNull @PolySigned Object[] es, int i);

    @Positive
    public void addFirst(@GuardSatisfied ArrayDeque<E> this, E e);

    @Positive
    public void addLast(@GuardSatisfied ArrayDeque<E> this, E e);

    @Positive
    public boolean addAll(Collection<? extends E> c);

    @Positive
    public boolean offerFirst(E e);

    @Positive
    public boolean offerLast(E e);

    @Positive
    public E removeFirst(@GuardSatisfied @NonEmpty @Shrinkable ArrayDeque<E> this);

    @Positive
    public E removeLast(@GuardSatisfied @NonEmpty @Shrinkable ArrayDeque<E> this);

    @Positive
    @Nullable
    @Positive
    public E pollFirst(@GuardSatisfied @Shrinkable ArrayDeque<E> this);

    @Positive
    @Nullable
    @Positive
    public E pollLast(@GuardSatisfied @Shrinkable ArrayDeque<E> this);

    @Positive
    public E getFirst(@GuardSatisfied @NonEmpty ArrayDeque<E> this);

    @Positive
    public E getLast(@GuardSatisfied @NonEmpty ArrayDeque<E> this);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public E peekFirst();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public E peekLast();

    @Positive
    public boolean removeFirstOccurrence(@GuardSatisfied @Shrinkable ArrayDeque<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public boolean removeLastOccurrence(@GuardSatisfied @Shrinkable ArrayDeque<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public boolean add(@GuardSatisfied ArrayDeque<E> this, E e);

    @Positive
    public boolean offer(@GuardSatisfied ArrayDeque<E> this, E e);

    @Positive
    public E remove(@GuardSatisfied @NonEmpty @Shrinkable ArrayDeque<E> this);

    @Positive
    @Nullable
    @Positive
    public E poll(@GuardSatisfied @Shrinkable ArrayDeque<E> this);

    @Positive
    public E element(@GuardSatisfied @NonEmpty ArrayDeque<E> this);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public E peek();

    @Positive
    public void push(@GuardSatisfied ArrayDeque<E> this, E e);

    @Positive
    public E pop(@GuardSatisfied @NonEmpty @Shrinkable ArrayDeque<E> this);

    @Positive
    boolean delete(int i);

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int size(@GuardSatisfied ArrayDeque<E> this);

    @Positive
    @EnsuresNonNullIf(expression = { "peek()", "peekFirst()", "peekLast()", "poll()", "pollFirst()", "pollLast()" }, result = false)
    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty(@GuardSatisfied ArrayDeque<E> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty ArrayDeque<E> this);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> descendingIterator(@PolyGrowShrink @PolyNonEmpty ArrayDeque<E> this);

    @Positive
    private class DeqIterator implements Iterator<E> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public final boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty DeqIterator this);

    @Positive
        void postDelete(boolean leftShifted);

    @Positive
        public final void remove();

    @Positive
        public void forEachRemaining(Consumer<? super E> action);
    @Positive
    }

    @Positive
    private class DescendingIterator extends DeqIterator {

    @Positive
        public final E next(@NonEmpty DescendingIterator this);

    @Positive
        void postDelete(boolean leftShifted);

    @Positive
        public final void forEachRemaining(Consumer<? super E> action);
    @Positive
    }

    @Positive
    public Spliterator<E> spliterator();

    @Positive
    final class DeqSpliterator implements Spliterator<E> {

    @Positive
        public DeqSpliterator trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        public boolean tryAdvance(Consumer<? super E> action);

    @Positive
        public long estimateSize();

    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    public void forEach(Consumer<? super E> action);

    @Positive
    public boolean removeIf(@Shrinkable ArrayDeque<E> this, Predicate<? super E> filter);

    @Positive
    public boolean removeAll(@Shrinkable ArrayDeque<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    public boolean retainAll(@GuardSatisfied @Shrinkable ArrayDeque<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied ArrayDeque<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public boolean remove(@GuardSatisfied @Shrinkable ArrayDeque<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public void clear(@GuardSatisfied @Shrinkable ArrayDeque<E> this);

    @Positive
    @SideEffectFree
    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public Object[] toArray(ArrayDeque<@PolyNull @PolySigned E> this);

    @Positive
    @SideEffectFree
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public <T> T[] toArray(@PolyNull T[] a);

    @Positive
    @SideEffectFree
    @Positive
    public ArrayDeque<E> clone(@GuardSatisfied ArrayDeque<E> this);

    @Positive
    void checkInvariants();
    @Positive
}
