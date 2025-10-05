/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.AbstractCollection;
import java.util.ListIterator;
import java.util.NoSuchElementException;
import java.util.function.Function;
import java.util.stream.Collector;

public class List<A> extends AbstractCollection<A> implements java.util.List<A> {

    public A head;

    public List<A> tail;

    @SuppressWarnings("unchecked")
    public static <A> List<A> nil();

    public static <A> List<A> filter(List<A> l, A elem);

    public List<A> intersect(List<A> that);

    public List<A> diff(List<A> that);

    public List<A> take(int n);

    public static <A> List<A> of(A x1);

    public static <A> List<A> of(A x1, A x2);

    public static <A> List<A> of(A x1, A x2, A x3);

    @SuppressWarnings({ "varargs", "unchecked" })
    public static <A> List<A> of(A x1, A x2, A x3, A... rest);

    public static <A> List<A> from(A[] array);

    public static <A> List<A> from(Iterable<? extends A> coll);

    @Deprecated
    public static <A> List<A> fill(int len, A init);

    @Override
    public boolean isEmpty();

    public boolean nonEmpty();

    public int length();

    @Override
    public int size();

    public List<A> setTail(List<A> tail);

    public List<A> prepend(A x);

    public List<A> prependList(List<A> xs);

    public List<A> reverse();

    public List<A> append(A x);

    public List<A> appendList(List<A> x);

    public List<A> appendList(ListBuffer<A> x);

    @Override
    @SuppressWarnings("unchecked")
    public <T> T[] toArray(T[] vec);

    public Object[] toArray();

    public String toString(String sep);

    @Override
    public String toString();

    @Override
    public int hashCode();

    @Override
    public boolean equals(Object other);

    public static boolean equals(List<?> xs, List<?> ys);

    @Override
    @Pure
    public boolean contains(Object x);

    public A last();

    @SuppressWarnings("unchecked")
    public <Z> List<Z> map(Function<A, Z> mapper);

    @SuppressWarnings("unchecked")
    public static <T> List<T> convert(Class<T> klass, List<?> list);

    @Override
    public Iterator<A> iterator();

    public A get(int index);

    public boolean addAll(int index, Collection<? extends A> c);

    public A set(int index, A element);

    public void add(int index, A element);

    public A remove(int index);

    public int indexOf(Object o);

    public int lastIndexOf(Object o);

    public ListIterator<A> listIterator();

    public ListIterator<A> listIterator(int index);

    public java.util.List<A> subList(int fromIndex, int toIndex);

    public static <Z> Collector<Z, ListBuffer<Z>, List<Z>> collector();
}
