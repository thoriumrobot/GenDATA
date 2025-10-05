/*
    @Positive
 * Copyright (c) 2005, 2012, Oracle and/or its affiliates. All rights reserved.
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
package java.net;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.net.URI;
    @Positive
import java.net.CookieStore;
    @Positive
import java.net.HttpCookie;
    @Positive
import java.net.URISyntaxException;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.concurrent.locks.ReentrantLock;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
class InMemoryCookieStore implements CookieStore {

    @Positive
    public InMemoryCookieStore() {
    @Positive
    }

    @Positive
    public void add(URI uri, HttpCookie cookie);

    @Positive
    public List<HttpCookie> get(URI uri);

    @Positive
    public List<HttpCookie> getCookies();

    @Positive
    public List<URI> getURIs();

    @Positive
    public boolean remove(URI uri, HttpCookie ck);

    @Positive
    public boolean removeAll();
    @Positive
}

// CFWR semantic augmentation - variant 1
