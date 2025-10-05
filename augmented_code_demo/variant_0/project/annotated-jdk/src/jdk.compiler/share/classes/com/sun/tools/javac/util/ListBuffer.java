/*
    @Positive
 * Copyright (c) 1999, 2013, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.AbstractQueue;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.NoSuchElementException;

    @Positive
public class ListBuffer<A> extends AbstractQueue<A> {

    @Positive
    public static <T> ListBuffer<T> of(T x);

    @Positive
    public ListBuffer() {
    @Positive
    }

    @Positive
    public final void clear();

    @Positive
    public int length();

    @Positive
    public int size();

    @Positive
    public boolean isEmpty();

    @Positive
    public boolean nonEmpty();

    @Positive
    public ListBuffer<A> prepend(A x);

    @Positive
    public ListBuffer<A> append(A x);

    @Positive
    public ListBuffer<A> appendList(List<A> xs);

    @Positive
    public ListBuffer<A> appendList(ListBuffer<A> xs);

    @Positive
    public ListBuffer<A> appendArray(A[] xs);

    @Positive
    public List<A> toList();

    @Positive
    @Pure
    @Positive
    public boolean contains(Object x);

    @Positive
    public <T> T[] toArray(T[] vec);

    @Positive
    public Object[] toArray();

    @Positive
    public A first();

    @Positive
    public A next();

    @Positive
    public Iterator<A> iterator();

    @Positive
    public boolean add(A a);

    @Positive
    public boolean remove(Object o);

    @Positive
    @Pure
    @Positive
    public boolean containsAll(Collection<?> c);

    @Positive
    public boolean addAll(Collection<? extends A> c);

    @Positive
    public boolean removeAll(Collection<?> c);

    @Positive
    public boolean retainAll(Collection<?> c);

    @Positive
    public boolean offer(A a);

    @Positive
    public A poll();

    @Positive
    public A peek();

    @Positive
    public A last();
    @Positive
}

// CFWR semantic augmentation - variant 0
