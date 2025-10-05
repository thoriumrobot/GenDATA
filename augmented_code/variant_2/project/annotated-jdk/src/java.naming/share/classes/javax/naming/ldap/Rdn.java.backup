/*
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.naming.ldap;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Collections;
import javax.naming.InvalidNameException;
import javax.naming.directory.BasicAttributes;
import javax.naming.directory.Attributes;
import javax.naming.directory.Attribute;
import javax.naming.NamingEnumeration;
import javax.naming.NamingException;
import java.io.Serializable;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.io.IOException;

public class Rdn implements Serializable, Comparable<Object> {

    public Rdn(Attributes attrSet) throws InvalidNameException {
    }

    public Rdn(String rdnString) throws InvalidNameException {
    }

    public Rdn(Rdn rdn) {
    }

    public Rdn(String type, Object value) throws InvalidNameException {
    }

    Rdn put(String type, Object value);

    void sort();

    public Object getValue();

    public String getType();

    public String toString();

    public int compareTo(Object obj);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public Attributes toAttributes();

    private static class RdnEntry implements Comparable<RdnEntry> {

        String getType();

        Object getValue();

        public int compareTo(RdnEntry that);

        public boolean equals(Object obj);

        public int hashCode();

        public String toString();
    }

    public int size();

    public static String escapeValue(Object val);

    public static Object unescapeValue(String val);
}
