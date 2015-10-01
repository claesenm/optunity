macro nogc(ex)
  quote
    local val
    try
      gc_enable(false)
      val = $(esc(ex))
    finally
      gc_enable(true)
    end
    val
  end
end