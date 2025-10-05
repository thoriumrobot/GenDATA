/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1998, 2018, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.KeyFor;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;

    @Positive
@CFComment({ "lock/nullness: Subclasses of this interface/class may opt to prohibit null elements" })
    @Positive
@AnnotatedFor({ "lock", "nullness" })
    @Positive
public interface SortedMap<K, V> extends Map<K, V> {

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    Comparator<? super K> comparator(@GuardSatisfied SortedMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    SortedMap<K, V> subMap(@GuardSatisfied SortedMap<K, V> this, @GuardSatisfied K fromKey, @GuardSatisfied K toKey);

    @Positive
    @SideEffectFree
    @Positive
    SortedMap<K, V> headMap(@GuardSatisfied SortedMap<K, V> this, K toKey);

    @Positive
    @SideEffectFree
    @Positive
    SortedMap<K, V> tailMap(@GuardSatisfied SortedMap<K, V> this, K fromKey);

    @Positive
    @SideEffectFree
    @Positive
    @KeyFor("this")
    @Positive
    K firstKey(@GuardSatisfied @NonEmpty SortedMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    @KeyFor("this")
    @Positive
    K lastKey(@GuardSatisfied @NonEmpty SortedMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    Set<@KeyFor({ "this" }) K> keySet(@GuardSatisfied SortedMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    Collection<V> values(@GuardSatisfied SortedMap<K, V> this);

    @Positive
    @SideEffectFree
    @Positive
    Set<Map.Entry<@KeyFor({ "this" }) K, V>> entrySet(@GuardSatisfied SortedMap<K, V> this);
    @Positive
}
