/*
    @Positive
 * Copyright (c) 2003, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
@AnnotatedFor({ "nullness", "index" })
    @Positive
public abstract class EnumSet<E extends Enum<E>> extends AbstractSet<E> implements Cloneable, java.io.Serializable {

    @Positive
    public static <E extends Enum<E>> EnumSet<E> noneOf(Class<E> elementType);

    @Positive
    public static <E extends Enum<E>> EnumSet<E> allOf(Class<E> elementType);

    @Positive
    abstract void addAll();

    @Positive
    public static <E extends Enum<E>> EnumSet<E> copyOf(EnumSet<E> s);

    @Positive
    public static <E extends Enum<E>> EnumSet<E> copyOf(Collection<E> c);

    @Positive
    public static <E extends Enum<E>> EnumSet<E> complementOf(EnumSet<E> s);

    @Positive
    public static <E extends Enum<E>> EnumSet<E> of(E e);

    @Positive
    public static <E extends Enum<E>> EnumSet<E> of(E e1, E e2);

    @Positive
    public static <E extends Enum<E>> EnumSet<E> of(E e1, E e2, E e3);

    @Positive
    public static <E extends Enum<E>> EnumSet<E> of(E e1, E e2, E e3, E e4);

    @Positive
    public static <E extends Enum<E>> EnumSet<E> of(E e1, E e2, E e3, E e4, E e5);

    @Positive
    @SafeVarargs
    @Positive
    public static <E extends Enum<E>> EnumSet<E> of(E first, E... rest);

    @Positive
    public static <E extends Enum<E>> EnumSet<E> range(E from, E to);

    @Positive
    abstract void addRange(E from, E to);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public EnumSet<E> clone();

    @Positive
    abstract void complement();

    @Positive
    final void typeCheck(E e);

    @Positive
    private static class SerializationProxy<E extends Enum<E>> implements java.io.Serializable {
    @Positive
    }

    @Positive
    @java.io.Serial
    @Positive
    Object writeReplace();
    @Positive
}

// CFWR semantic augmentation - variant 0
