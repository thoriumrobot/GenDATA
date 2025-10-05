/*
 * Copyright (c) 1999, 2013, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.util;

import org.checkerframework.dataflow.qual.Pure;
import java.util.AbstractQueue;
import java.util.Collection;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class ListBuffer<A> extends AbstractQueue<A> {

    public static <T> ListBuffer<T> of(T x);

    public ListBuffer() {
    }

    public final void clear();

    public int length();

    public int size();

    public boolean isEmpty();

    public boolean nonEmpty();

    public ListBuffer<A> prepend(A x);

    public ListBuffer<A> append(A x);

    public ListBuffer<A> appendList(List<A> xs);

    public ListBuffer<A> appendList(ListBuffer<A> xs);

    public ListBuffer<A> appendArray(A[] xs);

    public List<A> toList();

    @Pure
    public boolean contains(Object x);

    public <T> T[] toArray(T[] vec);

    public Object[] toArray();

    public A first();

    public A next();

    public Iterator<A> iterator();

    public boolean add(A a);

    public boolean remove(Object o);

    @Pure
    public boolean containsAll(Collection<?> c);

    public boolean addAll(Collection<? extends A> c);

    public boolean removeAll(Collection<?> c);

    public boolean retainAll(Collection<?> c);

    public boolean offer(A a);

    public A poll();

    public A peek();

    public A last();
}
