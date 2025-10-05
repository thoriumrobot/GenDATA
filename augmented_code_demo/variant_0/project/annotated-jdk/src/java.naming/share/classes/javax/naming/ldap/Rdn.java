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
package javax.naming.ldap;

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
import java.util.Iterator;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Collections;
    @Positive
import javax.naming.InvalidNameException;
    @Positive
import javax.naming.directory.BasicAttributes;
    @Positive
import javax.naming.directory.Attributes;
    @Positive
import javax.naming.directory.Attribute;
    @Positive
import javax.naming.NamingEnumeration;
    @Positive
import javax.naming.NamingException;
    @Positive
import java.io.Serializable;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.IOException;

    @Positive
public class Rdn implements Serializable, Comparable<Object> {

    @Positive
    public Rdn(Attributes attrSet) throws InvalidNameException {
    @Positive
    }

    @Positive
    public Rdn(String rdnString) throws InvalidNameException {
    @Positive
    }

    @Positive
    public Rdn(Rdn rdn) {
    @Positive
    }

    @Positive
    public Rdn(String type, Object value) throws InvalidNameException {
    @Positive
    }

    @Positive
    Rdn put(String type, Object value);

    @Positive
    void sort();

    @Positive
    public Object getValue();

    @Positive
    public String getType();

    @Positive
    public String toString();

    @Positive
    public int compareTo(Object obj);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public Attributes toAttributes();

    @Positive
    private static class RdnEntry implements Comparable<RdnEntry> {

    @Positive
        String getType();

    @Positive
        Object getValue();

    @Positive
        public int compareTo(RdnEntry that);

    @Positive
        public boolean equals(Object obj);

    @Positive
        public int hashCode();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public int size();

    @Positive
    public static String escapeValue(Object val);

    @Positive
    public static Object unescapeValue(String val);
    @Positive
}

// CFWR semantic augmentation - variant 0
