/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.reflect;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.signedness.qual.SignedPositive;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.StringJoiner;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class Modifier {

    public static boolean isPublic(int mod);

    public static boolean isPrivate(int mod);

    public static boolean isProtected(int mod);

    public static boolean isStatic(int mod);

    public static boolean isFinal(int mod);

    public static boolean isSynchronized(int mod);

    public static boolean isVolatile(int mod);

    public static boolean isTransient(int mod);

    public static boolean isNative(int mod);

    public static boolean isInterface(int mod);

    public static boolean isAbstract(int mod);

    public static boolean isStrict(int mod);

    public static String toString(int mod);

    @SignedPositive
    public static final int PUBLIC;

    @SignedPositive
    public static final int PRIVATE;

    @SignedPositive
    public static final int PROTECTED;

    @SignedPositive
    public static final int STATIC;

    @SignedPositive
    public static final int FINAL;

    @SignedPositive
    public static final int SYNCHRONIZED;

    @SignedPositive
    public static final int VOLATILE;

    @SignedPositive
    public static final int TRANSIENT;

    @SignedPositive
    public static final int NATIVE;

    @SignedPositive
    public static final int INTERFACE;

    @SignedPositive
    public static final int ABSTRACT;

    @SignedPositive
    public static final int STRICT;

    static boolean isSynthetic(int mod);

    static boolean isMandated(int mod);

    public static int classModifiers();

    public static int interfaceModifiers();

    public static int constructorModifiers();

    public static int methodModifiers();

    public static int fieldModifiers();

    public static int parameterModifiers();
}
