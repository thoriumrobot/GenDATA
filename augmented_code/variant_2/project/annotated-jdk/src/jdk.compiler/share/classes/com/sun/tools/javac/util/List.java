/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.lang.reflect.Array;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.AbstractCollection;
    @Positive
import java.util.ListIterator;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.stream.Collector;

    @Positive
public class List<A> extends AbstractCollection<A> implements java.util.List<A> {

    @Positive
    public A head;

    @Positive
    public List<A> tail;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <A> List<A> nil();

    @Positive
    public static <A> List<A> filter(List<A> l, A elem);

    @Positive
    public List<A> intersect(List<A> that);

    @Positive
    public List<A> diff(List<A> that);

    @Positive
    public List<A> take(int n);

    @Positive
    public static <A> List<A> of(A x1);

    @Positive
    public static <A> List<A> of(A x1, A x2);

    @Positive
    public static <A> List<A> of(A x1, A x2, A x3);

    @Positive
    @SuppressWarnings({ "varargs", "unchecked" })
    @Positive
    public static <A> List<A> of(A x1, A x2, A x3, A... rest);

    @Positive
    public static <A> List<A> from(A[] array);

    @Positive
    public static <A> List<A> from(Iterable<? extends A> coll);

    @Positive
    @Deprecated
    @Positive
    public static <A> List<A> fill(int len, A init);

    @Positive
    @Override
    @Positive
    public boolean isEmpty();

    @Positive
    public boolean nonEmpty();

    @Positive
    public int length();

    @Positive
    @Override
    @Positive
    public int size();

    @Positive
    public List<A> setTail(List<A> tail);

    @Positive
    public List<A> prepend(A x);

    @Positive
    public List<A> prependList(List<A> xs);

    @Positive
    public List<A> reverse();

    @Positive
    public List<A> append(A x);

    @Positive
    public List<A> appendList(List<A> x);

    @Positive
    public List<A> appendList(ListBuffer<A> x);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public <T> T[] toArray(T[] vec);

    @Positive
    public Object[] toArray();

    @Positive
    public String toString(String sep);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public boolean equals(Object other);

    @Positive
    public static boolean equals(List<?> xs, List<?> ys);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean contains(Object x);

    @Positive
    public A last();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public <Z> List<Z> map(Function<A, Z> mapper);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static <T> List<T> convert(Class<T> klass, List<?> list);

    @Positive
    @Override
    @Positive
    public Iterator<A> iterator();

    @Positive
    public A get(int index);

    @Positive
    public boolean addAll(int index, Collection<? extends A> c);

    @Positive
    public A set(int index, A element);

    @Positive
    public void add(int index, A element);

    @Positive
    public A remove(int index);

    @Positive
    public int indexOf(Object o);

    @Positive
    public int lastIndexOf(Object o);

    @Positive
    public ListIterator<A> listIterator();

    @Positive
    public ListIterator<A> listIterator(int index);

    @Positive
    public java.util.List<A> subList(int fromIndex, int toIndex);

    @Positive
    public static <Z> Collector<Z, ListBuffer<Z>, List<Z>> collector();
    @Positive
}
