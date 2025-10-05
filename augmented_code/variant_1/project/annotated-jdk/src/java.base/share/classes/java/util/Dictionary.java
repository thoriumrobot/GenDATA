/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresKeyFor;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;

    @Positive
@CFComment({ "lock/nullness: Subclasses of this interface/class may opt to prohibit null elements" })
    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public abstract class Dictionary<K, V> {

    @Positive
    public Dictionary() {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public abstract int size(@GuardSatisfied Dictionary<K, V> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public abstract boolean isEmpty(@GuardSatisfied Dictionary<K, V> this);

    @Positive
    public abstract Enumeration<K> keys();

    @Positive
    public abstract Enumeration<V> elements();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public abstract V get(@GuardSatisfied Dictionary<K, V> this, @UnknownSignedness @GuardSatisfied Object key);

    @Positive
    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Positive
    @Nullable
    @Positive
    public abstract V put(@GuardSatisfied Dictionary<K, V> this, K key, V value);

    @Positive
    @Nullable
    @Positive
    public abstract V remove(@GuardSatisfied Dictionary<K, V> this, Object key);
    @Positive
}
