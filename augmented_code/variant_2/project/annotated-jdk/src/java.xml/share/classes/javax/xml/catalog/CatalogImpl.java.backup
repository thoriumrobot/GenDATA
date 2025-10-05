/*
 * Copyright (c) 2015, 2017, Oracle and/or its affiliates. All rights reserved.
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
package javax.xml.catalog;

import com.sun.org.apache.xerces.internal.jaxp.SAXParserFactoryImpl;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import static javax.xml.catalog.BaseEntry.CatalogEntryType;
import static javax.xml.catalog.CatalogFeatures.DEFER_TRUE;
import javax.xml.catalog.CatalogFeatures.Feature;
import static javax.xml.catalog.CatalogMessages.formatMessage;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.xml.sax.SAXException;

class CatalogImpl extends GroupEntry implements Catalog {

    public CatalogImpl(CatalogFeatures f, URI... uris) throws CatalogException {
    }

    public CatalogImpl(CatalogImpl parent, CatalogFeatures f, URI... uris) throws CatalogException {
    }

    void load();

    @Override
    public void reset();

    boolean isTop();

    public Catalog getParent();

    public final void setDeferred(String value);

    public boolean isDeferred();

    public final void setResolve(String value);

    public final ResolveType getResolve();

    void markAsSearched();

    public boolean isEmpty();

    @Override
    public Stream<Catalog> catalogs();

    void addNextCatalog(NextCatalog catalog);

    void loadNextCatalogs();

    Catalog getCatalog(CatalogImpl parent, URI uri);

    void saveLoadedCatalog(String catalogId, CatalogImpl c);

    int loadedCatalogCount();
}
