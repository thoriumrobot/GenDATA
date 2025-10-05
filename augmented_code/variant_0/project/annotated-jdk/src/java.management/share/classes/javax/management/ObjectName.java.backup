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
package javax.management;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import com.sun.jmx.mbeanserver.GetPropertyAction;
import com.sun.jmx.mbeanserver.Util;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamField;
import java.security.AccessController;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;

@SuppressWarnings("serial")
public class ObjectName implements Comparable<ObjectName>, QueryExp {

    private static class Property {

        void setKeyIndex(int key_index);

        String getKeyString(String name);

        String getValueString(String name);
    }

    private static class PatternProperty extends Property {
    }

    public static ObjectName getInstance(String name) throws MalformedObjectNameException, NullPointerException;

    public static ObjectName getInstance(String domain, String key, String value) throws MalformedObjectNameException;

    public static ObjectName getInstance(String domain, Hashtable<String, String> table) throws MalformedObjectNameException;

    public static ObjectName getInstance(ObjectName name);

    public ObjectName(String name) throws MalformedObjectNameException {
    }

    public ObjectName(String domain, String key, String value) throws MalformedObjectNameException {
    }

    public ObjectName(String domain, Hashtable<String, String> table) throws MalformedObjectNameException {
    }

    public boolean isPattern();

    public boolean isDomainPattern();

    public boolean isPropertyPattern();

    public boolean isPropertyListPattern();

    public boolean isPropertyValuePattern();

    public boolean isPropertyValuePattern(String property);

    public String getCanonicalName();

    public String getDomain();

    public String getKeyProperty(String property);

    public Hashtable<String, String> getKeyPropertyList();

    public String getKeyPropertyListString();

    public String getCanonicalKeyPropertyListString();

    @Override
    public String toString();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object object);

    @Override
    public int hashCode();

    public static String quote(String s);

    public static String unquote(String q);

    public static final ObjectName WILDCARD;

    public boolean apply(ObjectName name);

    public void setMBeanServer(MBeanServer mbs);

    public int compareTo(ObjectName name);
}
