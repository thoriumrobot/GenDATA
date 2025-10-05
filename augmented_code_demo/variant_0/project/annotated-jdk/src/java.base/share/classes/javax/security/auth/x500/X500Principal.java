/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.security.auth.x500;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.security.Principal;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Map;
    @Positive
import sun.security.x509.X500Name;
    @Positive
import sun.security.util.*;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public final class X500Principal implements Principal, java.io.Serializable {

    @Positive
    @Interned
    @Positive
    public static final String RFC1779;

    @Positive
    @Interned
    @Positive
    public static final String RFC2253;

    @Positive
    @Interned
    @Positive
    public static final String CANONICAL;

    @Positive
    public X500Principal(String name) {
    @Positive
    }

    @Positive
    public X500Principal(String name, Map<String, String> keywordMap) {
    @Positive
    }

    @Positive
    public X500Principal(byte[] name) {
    @Positive
    }

    @Positive
    public X500Principal(InputStream is) {
    @Positive
    }

    @Positive
    public String getName();

    @Positive
    public String getName(String format);

    @Positive
    public String getName(String format, Map<String, String> oidMap);

    @Positive
    public byte[] getEncoded();

    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public int hashCode();
    @Positive
}

// CFWR semantic augmentation - variant 0
