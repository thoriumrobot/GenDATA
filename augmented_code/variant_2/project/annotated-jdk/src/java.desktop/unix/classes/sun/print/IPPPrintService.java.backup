/*
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.print;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.awt.GraphicsEnvironment;
import java.awt.Toolkit;
import javax.print.attribute.*;
import javax.print.attribute.standard.*;
import javax.print.DocFlavor;
import javax.print.DocPrintJob;
import javax.print.PrintService;
import javax.print.ServiceUIFactory;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.Date;
import java.util.Arrays;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import javax.print.event.PrintServiceAttributeListener;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLConnection;
import java.net.HttpURLConnection;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.DataInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ByteArrayInputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.Iterator;
import java.util.HashSet;
import java.util.Map;

public class IPPPrintService implements PrintService, SunPrinterJobService {

    public static final boolean debugPrint;

    protected static void debug_println(String str);

    public static final String OP_GET_ATTRIBUTES;

    public static final String OP_CUPS_GET_DEFAULT;

    public static final String OP_CUPS_GET_PRINTERS;

    public DocPrintJob createPrintJob();

    public synchronized Object getSupportedAttributeValues(Class<? extends Attribute> category, DocFlavor flavor, AttributeSet attributes);

    @SuppressWarnings("serial")
    private class ExtFinishing extends Finishings {

        EnumSyntax[] getAll();
    }

    public AttributeSet getUnsupportedAttributes(DocFlavor flavor, AttributeSet attributes);

    public synchronized DocFlavor[] getSupportedDocFlavors();

    public boolean isDocFlavorSupported(DocFlavor flavor);

    public CustomMediaSizeName findCustomMedia(MediaSizeName media);

    public synchronized Class<?>[] getSupportedAttributeCategories();

    public boolean isAttributeCategorySupported(Class<? extends Attribute> category);

    @SuppressWarnings("unchecked")
    public synchronized <T extends PrintServiceAttribute> T getAttribute(Class<T> category);

    public synchronized PrintServiceAttributeSet getAttributes();

    public boolean isIPPSupportedImages(String mimeType);

    public boolean isAttributeValueSupported(Attribute attr, DocFlavor flavor, AttributeSet attributes);

    public synchronized Object getDefaultAttributeValue(Class<? extends Attribute> category);

    public ServiceUIFactory getServiceUIFactory();

    public void wakeNotifier();

    public void addPrintServiceAttributeListener(PrintServiceAttributeListener listener);

    public void removePrintServiceAttributeListener(PrintServiceAttributeListener listener);

    String getDest();

    public String getName();

    public boolean usesClass(Class<?> c);

    public static HttpURLConnection getIPPConnection(URL url);

    public synchronized boolean isPostscript();

    public static boolean writeIPPRequest(OutputStream os, String operCode, AttributeClass[] attCl);

    public static HashMap<String, AttributeClass>[] readIPPResponse(InputStream inputStream);

    public String toString();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();
}
