/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.charset;

import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import jdk.internal.misc.VM;
import sun.nio.cs.ThreadLocalCoders;
import sun.security.action.GetPropertyAction;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.spi.CharsetProvider;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

public abstract class Charset implements Comparable<Charset> {

    private static class ExtendedProviderHolder {
    }

    public static boolean isSupported(String charsetName);

    public static Charset forName(String charsetName);

    @SuppressWarnings("removal")
    public static SortedMap<String, Charset> availableCharsets();

    public static Charset defaultCharset();

    protected Charset(String canonicalName, String[] aliases) {
    }

    public final String name();

    public final Set<String> aliases();

    public String displayName();

    public final boolean isRegistered();

    public String displayName(Locale locale);

    @Pure
    public abstract boolean contains(Charset cs);

    public abstract CharsetDecoder newDecoder();

    public abstract CharsetEncoder newEncoder();

    public boolean canEncode();

    public final CharBuffer decode(ByteBuffer bb);

    public final ByteBuffer encode(CharBuffer cb);

    public final ByteBuffer encode(String str);

    public final int compareTo(Charset that);

    public final int hashCode();

    public final boolean equals(Object ob);

    public final String toString();
}
