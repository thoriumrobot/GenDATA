/*
 * Copyright (c) 1995, 2020, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.util;

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nullness.qual.EnsuresKeyFor;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;

@CFComment({ "lock/nullness: Subclasses of this interface/class may opt to prohibit null elements" })
@AnnotatedFor({ "lock", "nullness", "index" })
public abstract class Dictionary<K, V> {

    public Dictionary() {
    }

    @Pure
    @NonNegative
    public abstract int size(@GuardSatisfied Dictionary<K, V> this);

    @Pure
    @EnsuresNonEmptyIf(result = false, expression = "this")
    public abstract boolean isEmpty(@GuardSatisfied Dictionary<K, V> this);

    public abstract Enumeration<K> keys();

    public abstract Enumeration<V> elements();

    @Pure
    @Nullable
    public abstract V get(@GuardSatisfied Dictionary<K, V> this, @UnknownSignedness @GuardSatisfied Object key);

    @EnsuresKeyFor(value = { "#1" }, map = { "this" })
    @Nullable
    public abstract V put(@GuardSatisfied Dictionary<K, V> this, K key, V value);

    @Nullable
    public abstract V remove(@GuardSatisfied Dictionary<K, V> this, Object key);
}
