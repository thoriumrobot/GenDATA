/*
 * Copyright (c) 2003, 2019, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import jdk.internal.access.SharedSecrets;

@AnnotatedFor({ "nullness", "index" })
public abstract class EnumSet<E extends Enum<E>> extends AbstractSet<E> implements Cloneable, java.io.Serializable {

    public static <E extends Enum<E>> EnumSet<E> noneOf(Class<E> elementType);

    public static <E extends Enum<E>> EnumSet<E> allOf(Class<E> elementType);

    abstract void addAll();

    public static <E extends Enum<E>> EnumSet<E> copyOf(EnumSet<E> s);

    public static <E extends Enum<E>> EnumSet<E> copyOf(Collection<E> c);

    public static <E extends Enum<E>> EnumSet<E> complementOf(EnumSet<E> s);

    public static <E extends Enum<E>> EnumSet<E> of(E e);

    public static <E extends Enum<E>> EnumSet<E> of(E e1, E e2);

    public static <E extends Enum<E>> EnumSet<E> of(E e1, E e2, E e3);

    public static <E extends Enum<E>> EnumSet<E> of(E e1, E e2, E e3, E e4);

    public static <E extends Enum<E>> EnumSet<E> of(E e1, E e2, E e3, E e4, E e5);

    @SafeVarargs
    public static <E extends Enum<E>> EnumSet<E> of(E first, E... rest);

    public static <E extends Enum<E>> EnumSet<E> range(E from, E to);

    abstract void addRange(E from, E to);

    @SuppressWarnings("unchecked")
    public EnumSet<E> clone();

    abstract void complement();

    final void typeCheck(E e);

    private static class SerializationProxy<E extends Enum<E>> implements java.io.Serializable {
    }

    @java.io.Serial
    Object writeReplace();
}
