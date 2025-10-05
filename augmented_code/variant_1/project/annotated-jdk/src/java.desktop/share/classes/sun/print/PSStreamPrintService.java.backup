/*
 * Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.
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
import java.io.OutputStream;
import java.util.Iterator;
import java.util.Locale;
import javax.print.DocFlavor;
import javax.print.DocPrintJob;
import javax.print.StreamPrintService;
import javax.print.StreamPrintServiceFactory;
import javax.print.ServiceUIFactory;
import javax.print.attribute.Attribute;
import javax.print.attribute.AttributeSet;
import javax.print.attribute.AttributeSetUtilities;
import javax.print.attribute.HashAttributeSet;
import javax.print.attribute.HashPrintServiceAttributeSet;
import javax.print.attribute.PrintServiceAttribute;
import javax.print.attribute.PrintServiceAttributeSet;
import javax.print.attribute.Size2DSyntax;
import javax.print.event.PrintServiceAttributeListener;
import javax.print.attribute.standard.JobName;
import javax.print.attribute.standard.RequestingUserName;
import javax.print.attribute.standard.Chromaticity;
import javax.print.attribute.standard.ColorSupported;
import javax.print.attribute.standard.Copies;
import javax.print.attribute.standard.CopiesSupported;
import javax.print.attribute.standard.Fidelity;
import javax.print.attribute.standard.Media;
import javax.print.attribute.standard.MediaPrintableArea;
import javax.print.attribute.standard.MediaSize;
import javax.print.attribute.standard.MediaSizeName;
import javax.print.attribute.standard.OrientationRequested;
import javax.print.attribute.standard.PageRanges;
import javax.print.attribute.standard.SheetCollate;
import javax.print.attribute.standard.Sides;

public class PSStreamPrintService extends StreamPrintService implements SunPrinterJobService {

    public PSStreamPrintService(OutputStream out) {
    }

    public String getOutputFormat();

    public DocFlavor[] getSupportedDocFlavors();

    public DocPrintJob createPrintJob();

    public boolean usesClass(Class<?> c);

    public String getName();

    public void addPrintServiceAttributeListener(PrintServiceAttributeListener listener);

    public void removePrintServiceAttributeListener(PrintServiceAttributeListener listener);

    public <T extends PrintServiceAttribute> T getAttribute(Class<T> category);

    public PrintServiceAttributeSet getAttributes();

    public boolean isDocFlavorSupported(DocFlavor flavor);

    public Class<?>[] getSupportedAttributeCategories();

    public boolean isAttributeCategorySupported(Class<? extends Attribute> category);

    public Object getDefaultAttributeValue(Class<? extends Attribute> category);

    public Object getSupportedAttributeValues(Class<? extends Attribute> category, DocFlavor flavor, AttributeSet attributes);

    public boolean isAttributeValueSupported(Attribute attr, DocFlavor flavor, AttributeSet attributes);

    public AttributeSet getUnsupportedAttributes(DocFlavor flavor, AttributeSet attributes);

    public ServiceUIFactory getServiceUIFactory();

    public String toString();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();
}
