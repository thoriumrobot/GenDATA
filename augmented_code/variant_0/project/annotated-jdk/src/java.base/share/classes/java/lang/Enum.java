/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.initialization.qual.UnknownInitialization;
    @Positive
import org.checkerframework.checker.nullness.qual.UnknownKeyFor;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.lock.qual.GuardedByUnknown;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.tainting.qual.Tainted;
    @Positive
import org.checkerframework.common.value.qual.PolyValue;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.Covariant;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectStreamException;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.constant.ClassDesc;
    @Positive
import java.lang.constant.Constable;
    @Positive
import java.lang.constant.ConstantDescs;
    @Positive
import java.lang.constant.DynamicConstantDesc;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.util.Optional;
    @Positive
import static java.util.Objects.requireNonNull;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index", "value", "tainting" })
    @Positive
@Covariant(0)
    @Positive
@SuppressWarnings("serial")
    @Positive
public abstract class Enum<E extends Enum<E>> implements Constable, Comparable<E>, Serializable {

    @Positive
    @Pure
    @Positive
    @PolyValue
    @Positive
    public final String name(@GuardedByUnknown @UnknownInitialization(java.lang.Enum.class) @PolyValue Enum<E> this);

    @Positive
    @NonNegative
    @Positive
    public final int ordinal();

    @Positive
    protected Enum(String name, @NonNegative int ordinal) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied Enum<E> this);

    @Positive
    @Pure
    @Positive
    public final boolean equals(@GuardSatisfied Enum<E> this, @GuardSatisfied @Nullable Object other);

    @Positive
    @Pure
    @Positive
    public final int hashCode(@GuardSatisfied Enum<E> this);

    @Positive
    @SideEffectFree
    @Positive
    protected final Object clone(@GuardSatisfied Enum<E> this) throws CloneNotSupportedException;

    @Positive
    @SuppressWarnings({ "rawtypes" })
    @Positive
    public final int compareTo(@UnknownKeyFor @Tainted E o);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public final Class<@Tainted E> getDeclaringClass();

    @Positive
    @Override
    @Positive
    public final Optional<EnumDesc<E>> describeConstable();

    @Positive
    @PolyValue
    @Positive
    public static <T extends Enum<T>> T valueOf(Class<T> enumClass, @PolyValue String name);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    protected final void finalize();

    @Positive
    public static final class EnumDesc<E extends Enum<E>> extends DynamicConstantDesc<E> {

    @Positive
        public static <E extends Enum<E>> EnumDesc<E> of(ClassDesc enumClass, String constantName);

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public E resolveConstantDesc(MethodHandles.Lookup lookup) throws ReflectiveOperationException;

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }
    @Positive
}
