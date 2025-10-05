/*
    @Positive
 * Copyright (c) 2015, 2017, Oracle and/or its affiliates. All rights reserved.
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
package javax.xml.catalog;

    @Positive
import com.sun.org.apache.xerces.internal.jaxp.SAXParserFactoryImpl;
    @Positive
import java.io.IOException;
    @Positive
import java.net.MalformedURLException;
    @Positive
import java.net.URI;
    @Positive
import java.net.URISyntaxException;
    @Positive
import java.net.URL;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;
    @Positive
import static javax.xml.catalog.BaseEntry.CatalogEntryType;
    @Positive
import static javax.xml.catalog.CatalogFeatures.DEFER_TRUE;
    @Positive
import javax.xml.catalog.CatalogFeatures.Feature;
    @Positive
import static javax.xml.catalog.CatalogMessages.formatMessage;
    @Positive
import javax.xml.parsers.ParserConfigurationException;
    @Positive
import javax.xml.parsers.SAXParser;
    @Positive
import javax.xml.parsers.SAXParserFactory;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.xml.sax.SAXException;

    @Positive
class CatalogImpl extends GroupEntry implements Catalog {

    @Positive
    public CatalogImpl(CatalogFeatures f, URI... uris) throws CatalogException {
    @Positive
    }

    @Positive
    public CatalogImpl(CatalogImpl parent, CatalogFeatures f, URI... uris) throws CatalogException {
    @Positive
    }

    @Positive
    void load();

    @Positive
    @Override
    @Positive
    public void reset();

    @Positive
    boolean isTop();

    @Positive
    public Catalog getParent();

    @Positive
    public final void setDeferred(String value);

    @Positive
    public boolean isDeferred();

    @Positive
    public final void setResolve(String value);

    @Positive
    public final ResolveType getResolve();

    @Positive
    void markAsSearched();

    @Positive
    public boolean isEmpty();

    @Positive
    @Override
    @Positive
    public Stream<Catalog> catalogs();

    @Positive
    void addNextCatalog(NextCatalog catalog);

    @Positive
    void loadNextCatalogs();

    @Positive
    Catalog getCatalog(CatalogImpl parent, URI uri);

    @Positive
    void saveLoadedCatalog(String catalogId, CatalogImpl c);

    @Positive
    int loadedCatalogCount();
    @Positive
}

// CFWR semantic augmentation - variant 0
