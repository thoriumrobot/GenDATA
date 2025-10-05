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
package sun.nio.cs.ext;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.spi.CharsetProvider;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.TreeMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;

    @Positive
public class AbstractCharsetProvider extends CharsetProvider {

    @Positive
    protected AbstractCharsetProvider() {
    @Positive
    }

    @Positive
    protected AbstractCharsetProvider(String pkgPrefixName) {
    @Positive
    }

    @Positive
    protected void charset(String name, String className, String[] aliases);

    @Positive
    protected void deleteCharset(String name, String[] aliases);

    @Positive
    protected boolean hasCharset(String name);

    @Positive
    protected void init();

    @Positive
    public final Charset charsetForName(String charsetName);

    @Positive
    public final Iterator<Charset> charsets();

    @Positive
    public final String[] aliases(String charsetName);
    @Positive
}

// CFWR semantic augmentation - variant 0
