/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package sun.print;

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
import java.awt.GraphicsEnvironment;
    @Positive
import java.awt.Toolkit;
    @Positive
import javax.print.attribute.*;
    @Positive
import javax.print.attribute.standard.*;
    @Positive
import javax.print.DocFlavor;
    @Positive
import javax.print.DocPrintJob;
    @Positive
import javax.print.PrintService;
    @Positive
import javax.print.ServiceUIFactory;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Date;
    @Positive
import java.util.Arrays;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import javax.print.event.PrintServiceAttributeListener;
    @Positive
import java.net.URI;
    @Positive
import java.net.URISyntaxException;
    @Positive
import java.net.URL;
    @Positive
import java.net.URLConnection;
    @Positive
import java.net.HttpURLConnection;
    @Positive
import java.io.File;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.OutputStreamWriter;
    @Positive
import java.io.DataInputStream;
    @Positive
import java.io.ByteArrayOutputStream;
    @Positive
import java.io.ByteArrayInputStream;
    @Positive
import java.io.BufferedReader;
    @Positive
import java.io.InputStreamReader;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Map;

    @Positive
public class IPPPrintService implements PrintService, SunPrinterJobService {

    @Positive
    public static final boolean debugPrint;

    @Positive
    protected static void debug_println(String str);

    @Positive
    public static final String OP_GET_ATTRIBUTES;

    @Positive
    public static final String OP_CUPS_GET_DEFAULT;

    @Positive
    public static final String OP_CUPS_GET_PRINTERS;

    @Positive
    public DocPrintJob createPrintJob();

    @Positive
    public synchronized Object getSupportedAttributeValues(Class<? extends Attribute> category, DocFlavor flavor, AttributeSet attributes);

    @Positive
    @SuppressWarnings("serial")
    @Positive
    private class ExtFinishing extends Finishings {

    @Positive
        EnumSyntax[] getAll();
    @Positive
    }

    @Positive
    public AttributeSet getUnsupportedAttributes(DocFlavor flavor, AttributeSet attributes);

    @Positive
    public synchronized DocFlavor[] getSupportedDocFlavors();

    @Positive
    public boolean isDocFlavorSupported(DocFlavor flavor);

    @Positive
    public CustomMediaSizeName findCustomMedia(MediaSizeName media);

    @Positive
    public synchronized Class<?>[] getSupportedAttributeCategories();

    @Positive
    public boolean isAttributeCategorySupported(Class<? extends Attribute> category);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public synchronized <T extends PrintServiceAttribute> T getAttribute(Class<T> category);

    @Positive
    public synchronized PrintServiceAttributeSet getAttributes();

    @Positive
    public boolean isIPPSupportedImages(String mimeType);

    @Positive
    public boolean isAttributeValueSupported(Attribute attr, DocFlavor flavor, AttributeSet attributes);

    @Positive
    public synchronized Object getDefaultAttributeValue(Class<? extends Attribute> category);

    @Positive
    public ServiceUIFactory getServiceUIFactory();

    @Positive
    public void wakeNotifier();

    @Positive
    public void addPrintServiceAttributeListener(PrintServiceAttributeListener listener);

    @Positive
    public void removePrintServiceAttributeListener(PrintServiceAttributeListener listener);

    @Positive
    String getDest();

    @Positive
    public String getName();

    @Positive
    public boolean usesClass(Class<?> c);

    @Positive
    public static HttpURLConnection getIPPConnection(URL url);

    @Positive
    public synchronized boolean isPostscript();

    @Positive
    public static boolean writeIPPRequest(OutputStream os, String operCode, AttributeClass[] attCl);

    @Positive
    public static HashMap<String, AttributeClass>[] readIPPResponse(InputStream inputStream);

    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();
    @Positive
}
