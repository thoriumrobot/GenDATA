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
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.reflect.Field;
    @Positive
import java.util.AbstractSet;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Map;
    @Positive
import java.util.NavigableSet;
    @Positive
import java.util.Set;
    @Positive
import java.util.SortedSet;
    @Positive
import java.util.Spliterator;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class ConcurrentSkipListSet<E extends @NonNull Object> extends AbstractSet<E> implements NavigableSet<E>, Cloneable, java.io.Serializable {

    @Positive
    public ConcurrentSkipListSet() {
    @Positive
    }

    @Positive
    public ConcurrentSkipListSet(@Nullable Comparator<? super E> comparator) {
    @Positive
    }

    @Positive
    public ConcurrentSkipListSet(Collection<? extends E> c) {
    @Positive
    }

    @Positive
    public ConcurrentSkipListSet(SortedSet<E> s) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public ConcurrentSkipListSet<E> clone();

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
    public boolean contains(@GuardSatisfied @UnknownSignedness Object o);

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public boolean add(E e);

    @Positive
    public boolean remove(@GuardSatisfied @UnknownSignedness Object o);

    @Positive
    public void clear();

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty ConcurrentSkipListSet<E> this);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> descendingIterator(@PolyGrowShrink @PolyNonEmpty ConcurrentSkipListSet<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public boolean removeAll(Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    public E lower(E e);

    @Positive
    public E floor(E e);

    @Positive
    public E ceiling(E e);

    @Positive
    public E higher(E e);

    @Positive
    @Nullable
    @Positive
    public E pollFirst();

    @Positive
    @Nullable
    @Positive
    public E pollLast();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Comparator<? super E> comparator();

    @Positive
    public E first(@NonEmpty ConcurrentSkipListSet<E> this);

    @Positive
    public E last(@NonEmpty ConcurrentSkipListSet<E> this);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public NavigableSet<E> subSet(@PolyGrowShrink @PolyNonEmpty ConcurrentSkipListSet<E> this, E fromElement, boolean fromInclusive, E toElement, boolean toInclusive);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public NavigableSet<E> headSet(@PolyGrowShrink @PolyNonEmpty ConcurrentSkipListSet<E> this, E toElement, boolean inclusive);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public NavigableSet<E> tailSet(@PolyGrowShrink @PolyNonEmpty ConcurrentSkipListSet<E> this, E fromElement, boolean inclusive);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public NavigableSet<E> subSet(@PolyGrowShrink @PolyNonEmpty ConcurrentSkipListSet<E> this, E fromElement, E toElement);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public NavigableSet<E> headSet(@PolyGrowShrink @PolyNonEmpty ConcurrentSkipListSet<E> this, E toElement);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public NavigableSet<E> tailSet(@PolyGrowShrink @PolyNonEmpty ConcurrentSkipListSet<E> this, E fromElement);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public NavigableSet<E> descendingSet(@PolyGrowShrink @PolyNonEmpty ConcurrentSkipListSet<E> this);

    @Positive
    @SuppressWarnings({ "unchecked" })
    @Positive
    @SideEffectFree
    @Positive
    public Spliterator<E> spliterator();
    @Positive
}

// CFWR semantic augmentation - variant 1
